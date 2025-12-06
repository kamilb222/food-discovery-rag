import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel, AutoModelForCausalLM
import os

# --- Page Config ---
st.set_page_config(
    page_title="Nutritionist Assistant",
    page_icon="🥗",
    layout="wide"
)

# --- Load Resources (Cached) ---
@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        df = pd.read_parquet(os.path.join(base_dir, 'project_data_clean.parquet'))
        image_ids = np.load(os.path.join(base_dir, 'models', 'image_ids.npy'), allow_pickle=True)
        text_index = faiss.read_index(os.path.join(base_dir, 'models', 'text_search.index'))
        image_index = faiss.read_index(os.path.join(base_dir, 'models', 'image_search.index'))
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        return None

    # 2. Load Models
    # Text Embedding
    text_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device)

    # Image Embedding
    vit_model_name = 'google/vit-base-patch16-224-in21k'
    vit_processor = ViTImageProcessor.from_pretrained(vit_model_name)
    vit_model = ViTModel.from_pretrained(vit_model_name).to(device)

    # Generative LLM
    llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, dtype="auto", device_map="auto")
    
    return {
        'df': df,
        'image_ids': image_ids,
        'text_index': text_index,
        'image_index': image_index,
        'device': device,
        'tokenizer': tokenizer,
        'text_model': text_model,
        'vit_processor': vit_processor,
        'vit_model': vit_model,
        'llm_tokenizer': llm_tokenizer,
        'llm_model': llm_model
    }

resources = load_resources()

if not resources:
    st.stop()

# --- Logic Functions ---

def get_text_embedding(text, resources):
    inputs = resources['tokenizer'](text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(resources['device'])
    with torch.no_grad():
        outputs = resources['text_model'](**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def get_image_embedding(image, resources):
    if isinstance(image, bytes):
         pass 
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = resources['vit_processor'](images=image, return_tensors="pt").to(resources['device'])
    with torch.no_grad():
        outputs = resources['vit_model'](**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def retrieve_products(query, image, resources, use_hybrid=False, k=5):
    scores = {}
    c = 60
    
    # --- STRATEGY: Hybrid vs Visual Priority ---
    
    # 1. Image Search
    if image is not None:
        query_vector = get_image_embedding(image, resources)
        distances, indices = resources['image_index'].search(query_vector, k * 2)
        retrieved_img_indices = indices[0]
        
        for rank, img_idx in enumerate(retrieved_img_indices):
            if img_idx < len(resources['image_ids']):
                code = resources['image_ids'][img_idx]
                matches = resources['df'].index[resources['df']['code'] == code].tolist()
                if matches:
                    df_idx = matches[0]
                    if df_idx not in scores: scores[df_idx] = 0
                    scores[df_idx] += 1 / (c + rank + 1)

    # 2. Text Search
    # Run if query exists AND (no image provided OR hybrid mode is ON)
    if query and (image is None or use_hybrid):
        query_vector = get_text_embedding(query, resources)
        distances, indices = resources['text_index'].search(query_vector, k * 2)
        text_indices = indices[0].tolist()
        
        for rank, idx in enumerate(text_indices):
            if idx not in scores: scores[idx] = 0
            scores[idx] += 1 / (c + rank + 1)
            
    # Sort by Score (Descending)
    sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Return top K results
    top_indices = sorted_indices[:k]
    return resources['df'].loc[top_indices]

def format_product_for_llm(row):
    parts = []
    name = row.get('product_name')
    if not name: return ""
    parts.append(f"Name: {name}")
    
    if pd.notna(row.get('energy-kcal_100g')):
        parts.append(f"Calories: {row['energy-kcal_100g']:.0f}kcal")
    if pd.notna(row.get('proteins_100g')):
        parts.append(f"Protein: {row['proteins_100g']:.1f}g")
    if pd.notna(row.get('fat_100g')):
        parts.append(f"Fat: {row['fat_100g']:.1f}g")
    if pd.notna(row.get('carbs_100g')):
        parts.append(f"Carbs: {row['carbs_100g']:.1f}g")
        
    ingr = str(row.get('ingredients_text', ''))
    if ingr and ingr != 'None' and ingr != 'nan':
        if len(ingr) > 100:
            ingr = ingr[:100] + "..."
        parts.append(f"Ingredients: {ingr}")
        
    return " | ".join(parts)

def check_safety_guardrails(user_query, products):
    """
    Scans retrieved products for dietary conflicts BEFORE the LLM answers.
    Returns a warning string if a conflict is found.
    """
    user_query = user_query.lower()
    
    # Define simple rules
    forbidden_ingredients = {
        'vegan': ['milk', 'dairy', 'egg', 'honey', 'meat', 'beef', 'pork', 'chicken', 'fish', 'gelatin', 'whey', 'casein'],
        'vegetarian': ['meat', 'beef', 'pork', 'chicken', 'fish', 'gelatin'],
        'gluten': ['wheat', 'barley', 'rye', 'malt', 'flour'],
        'peanut': ['peanut', 'nut']
    }
    
    warnings = []
    
    # Check which constraint the user is asking about
    active_constraint = None
    for constraint in forbidden_ingredients:
        if constraint in user_query:
            active_constraint = constraint
            break
            
    if not active_constraint:
        return None # No safety check needed
        
    # Scan the retrieved products
    bad_ingredients = forbidden_ingredients[active_constraint]
    
    for i, (_, row) in enumerate(products.iterrows()):
        ingredients = str(row.get('ingredients_text', '')).lower()
        product_name = str(row.get('product_name', ''))
        
        # Check if any forbidden ingredient is in the text
        found_conflicts = [bad for bad in bad_ingredients if bad in ingredients]
        
        if found_conflicts:
            warnings.append(
                f"⚠️ **SAFETY WARNING:** Product {i+1} ({product_name}) contains **{', '.join(found_conflicts)}**, so it is NOT {active_constraint}."
            )
            
    if warnings:
        return "\n\n".join(warnings)
    
    return None

def rewrite_query(user_query, history, resources):
    """
    Smartly rewrites the query. 
    It detects if the user is asking a follow-up (adds context) 
    OR starting a new topic (ignores context).
    """
    if not history:
        return user_query
        
    history_str = ""
    for msg in history[-2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 150: 
            content = content[:150] + "..."
        history_str += f"{role}: {content}\n"

    # We explicitly tell Qwen to detect topic shifts.
    system_instruction = (
        "You are a search query optimizer. Your job is to clarify the user's latest question.\n"
        "Rules:\n"
        "1. If the user refers to previous items (e.g., 'is it vegan?', 'how much fat?'), REWRITE the query to include the product names from history.\n"
        "2. If the user asks a completely NEW question (e.g., 'show me apples', 'I want pizza'), output the user's query EXACTLY as is. Do NOT add old context.\n"
        "3. Output ONLY the rewritten query text."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Chat History:\n{history_str}\n\nUser Input: {user_query}\n\nOptimized Query:"}
    ]
    
    text = resources['llm_tokenizer'].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = resources['llm_tokenizer']([text], return_tensors="pt").to(resources['llm_model'].device)
    
    generated_ids = resources['llm_model'].generate(
        **model_inputs,
        max_new_tokens=64,
        do_sample=False
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    rewritten_query = resources['llm_tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
    return rewritten_query.strip()
    
def generate_answer(user_query, products, resources):
    if products.empty:
        return "I couldn't find any relevant products to analyze."

    # 1. Build Context
    product_contexts = []
    for i, (_, row) in enumerate(products.iterrows()):
        clean_info = format_product_for_llm(row)
        if clean_info:
            product_contexts.append(f"Product {i+1}: {clean_info}")
    
    context_str = "\n".join(product_contexts)
    
    # 2. Optimized System Prompt
    system_message = (
        "You are an expert Nutritionist Assistant. "
        "Analyze the provided product data to answer the user's question. "
        "Guidelines:\n"
        "1. Answer based directly on the user's intent (e.g., if they ask for high carbs, highlight high carb items).\n"
        "2. Use general knowledge to fill gaps (e.g., Pizza = High Calorie, Pepperoni = Meat).\n"
        "3. Be honest: warn about high sugar/saturated fat if relevant to health, but don't preach."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Here is the product data:\n{context_str}\n\nQuestion: {user_query}"}
    ]
    
    # 3. Apply Chat Template
    text = resources['llm_tokenizer'].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 4. Generate
    model_inputs = resources['llm_tokenizer']([text], return_tensors="pt").to(resources['llm_model'].device)

    generated_ids = resources['llm_model'].generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    
    # 5. Decode
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = resources['llm_tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width='stretch')
    else:
        image = None

    # Hybrid Search Toggle
    use_hybrid = st.toggle("Use Text for Search?", value=False, help="If on, searches for matches to BOTH the image and your text. If off, prioritizes the image.")

# Main Chat Interface
st.title("🥗 Nutritionist Assistant")
st.markdown("Ask me about healthy snacks, high protein foods, or upload a food image to find similar items!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What are you looking for?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        search_query = rewrite_query(prompt, st.session_state.messages[:-1], resources)
        
        # 1. Retrieve
        retrieved_products = retrieve_products(search_query, image, resources, use_hybrid=use_hybrid)
        
        # 2. Safety Check
        safety_warning = check_safety_guardrails(search_query, retrieved_products)
        
        if safety_warning:
            final_answer = safety_warning + "\n\n(Note: The AI generation was skipped for safety reasons.)"
        else:
            final_answer = generate_answer(prompt, retrieved_products, resources)
        
        # Display Assistant Response
        with st.chat_message("assistant"):
            st.markdown(final_answer)
            
            # Display Top 3 Products
            st.subheader("Top Recommendations")
            cols = st.columns(3)
            
            for i, (_, row) in enumerate(retrieved_products.head(3).iterrows()):
                with cols[i]:
                    img_path = None
                    if 'code' in row:
                        code_str = str(row['code'])
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        possible_path = os.path.join(base_dir, "images", f"{code_str}.jpg")
                        if os.path.exists(possible_path):
                            img_path = possible_path
                    
                    if img_path:
                        st.image(img_path, width='stretch')
                    else:
                        st.text("Image not found")
                        
                    st.markdown(f"**{row.get('product_name', 'Unknown')}**")
                    
                    # Nutri-Score
                    ns = row.get('nutriscore_grade', '?').upper() if pd.notna(row.get('nutriscore_grade')) else '?'
                    st.caption(f"Nutri-Score: {ns}")
                    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
