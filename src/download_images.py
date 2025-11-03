"""
Image Download Script for Multi-Modal Food Discovery System

This script downloads product images from the Open Food Facts dataset.
It reads the cleaned dataset and downloads images from the constructed URLs.

Note: URLs are already constructed in Notebook 01 using the pattern:
https://images.openfoodfacts.org/images/products/{path}/1.400.jpg

Output: images/ directory with product images, then compressed to images.zip
"""

import pandas as pd
import numpy as np
import requests
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging
from typing import Tuple, Optional
import hashlib
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
# Get the root directory (one level up from 'src/')
ROOT_DIR = SCRIPT_DIR.parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT_DIR / 'image_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
IMAGE_DIR = ROOT_DIR / 'images'
PARQUET_FILE = ROOT_DIR / 'notebooks' / 'project_data_clean.parquet'
ZIP_FILE = ROOT_DIR / 'images.zip'
MAX_WORKERS = 10  # Number of concurrent downloads
TIMEOUT = 10  # Timeout for each download in seconds
MAX_RETRIES = 3  # Maximum retries for failed downloads
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for large files
MIN_IMAGE_SIZE = 1024  # Minimum valid image size (1KB)

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    logging.info(f"Created/verified directory: {IMAGE_DIR}")

def load_dataset() -> pd.DataFrame:
    """Load the cleaned dataset."""
    if not os.path.exists(PARQUET_FILE):
        logging.error(f"Dataset file not found: {PARQUET_FILE}")
        logging.error("Please run notebook 01_Data_Ingestion_Cleaning_EDA.ipynb first!")
        sys.exit(1)
    
    logging.info(f"Loading dataset from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    logging.info(f"Loaded {len(df):,} products")
    
    # Check if image_url column exists
    if 'image_url' not in df.columns:
        logging.error("'image_url' column not found in dataset!")
        logging.error("Please ensure Notebook 01 properly constructs image URLs.")
        sys.exit(1)
    
    # Filter to only products with valid URLs
    df_with_urls = df[df['image_url'].notna()].copy()
    logging.info(f"Found {len(df_with_urls):,} products with image URLs")
    
    return df_with_urls

def get_image_filename(code: str) -> str:
    """Generate a consistent filename for the image based on product code."""
    # Clean the code to ensure valid filename
    clean_code = str(code).replace('/', '_').replace('\\', '_')
    return f"{clean_code}.jpg"

def verify_image_integrity(filepath: str) -> bool:
    """Verify that the downloaded image is valid."""
    try:
        # Check file size
        if not os.path.exists(filepath):
            return False
        
        file_size = os.path.getsize(filepath)
        if file_size < MIN_IMAGE_SIZE:
            return False
        
        # Try to open with PIL to verify it's a valid image
        from PIL import Image
        with Image.open(filepath) as img:
            img.verify()
        
        return True
    except Exception:
        return False

def download_image(row: pd.Series, session: requests.Session) -> Tuple[str, str, bool]:
    """
    Download a single image from the constructed URL.
    
    Args:
        row: DataFrame row containing 'code' and 'image_url'
        session: Requests session for connection pooling
        
    Returns:
        Tuple of (code, status, success)
    """
    code = str(row['code'])
    url = row['image_url']
    
    # Check if URL is valid
    if pd.isna(url) or not isinstance(url, str) or not url.startswith('http'):
        return code, "invalid_url", False
    
    filename = get_image_filename(code)
    filepath = os.path.join(IMAGE_DIR, filename)
    
    # Skip if already downloaded and valid
    if os.path.exists(filepath):
        if verify_image_integrity(filepath):
            return code, "exists", True
        else:
            # Remove corrupted file
            os.remove(filepath)
    
    # Attempt download with retries
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=TIMEOUT, stream=True)
            
            if response.status_code == 200:
                # Download in chunks to handle large files
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                
                # Verify file was created and is valid
                if verify_image_integrity(filepath):
                    return code, "success", True
                else:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return code, "corrupted_image", False
                    
            elif response.status_code == 404:
                return code, "not_found", False
            else:
                if attempt == MAX_RETRIES - 1:
                    return code, f"http_{response.status_code}", False
                time.sleep(1)  # Wait before retry
                
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES - 1:
                return code, "timeout", False
            time.sleep(1)
            
        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES - 1:
                return code, "connection_error", False
            time.sleep(2)
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return code, f"error_{type(e).__name__}", False
            time.sleep(1)
    
    return code, "max_retries", False

def download_all_images(df: pd.DataFrame):
    """Download all images using thread pool."""
    # Filter to rows with valid URLs
    rows_to_download = []
    for _, row in df.iterrows():
        if pd.notna(row.get('image_url')) and row['image_url'].startswith('http'):
            rows_to_download.append(row)
    
    total_images = len(rows_to_download)
    
    if total_images == 0:
        logging.warning("No valid image URLs found in dataset!")
        return {'success': 0, 'exists': 0, 'failed': 0, 'errors': {}}
    
    logging.info(f"Starting download of {total_images:,} images...")
    logging.info(f"Using {MAX_WORKERS} concurrent workers")
    
    # Statistics
    stats = {
        'success': 0,
        'exists': 0,
        'failed': 0,
        'errors': {}
    }
    
    # Create a session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Progress bar
    with tqdm(total=total_images, desc="Downloading images", unit="img") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all download tasks
            future_to_row = {
                executor.submit(download_image, row, session): row 
                for row in rows_to_download
            }
            
            # Process completed downloads
            for future in as_completed(future_to_row):
                try:
                    code, status, success = future.result()
                    
                    if status == "success":
                        stats['success'] += 1
                    elif status == "exists":
                        stats['exists'] += 1
                    else:
                        stats['failed'] += 1
                        if status not in stats['errors']:
                            stats['errors'][status] = 0
                        stats['errors'][status] += 1
                    
                    pbar.update(1)
                    
                    # Update progress bar description with stats
                    pbar.set_postfix({
                        'Success': stats['success'],
                        'Exists': stats['exists'],
                        'Failed': stats['failed']
                    })
                except Exception as e:
                    logging.error(f"Error processing future: {e}")
                    stats['failed'] += 1
                    pbar.update(1)
    
    # Log final statistics
    logging.info("=" * 60)
    logging.info("DOWNLOAD COMPLETE")
    logging.info("=" * 60)
    logging.info(f" Successfully downloaded: {stats['success']:,}")
    logging.info(f" Already existed: {stats['exists']:,}")
    logging.info(f" Failed downloads: {stats['failed']:,}")
    
    if stats['errors']:
        logging.info("\nError breakdown:")
        for error_type, count in sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  • {error_type}: {count}")
    
    total_downloaded = stats['success'] + stats['exists']
    if total_images > 0:
        success_rate = (total_downloaded / total_images) * 100
        logging.info(f"\n Overall success rate: {success_rate:.1f}%")
    
    return stats

def create_zip_archive():
    """Create a zip file containing all downloaded images."""
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    
    if not image_files:
        logging.error("No images found to compress!")
        return False
    
    logging.info(f"\nCreating zip archive: {ZIP_FILE}")
    logging.info(f"Compressing {len(image_files):,} images...")
    
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, filename in enumerate(tqdm(image_files, desc="Compressing images")):
            filepath = os.path.join(IMAGE_DIR, filename)
            # Add file to zip with just the filename (no path)
            zipf.write(filepath, filename)
            
            # Log progress every 1000 files
            if (i + 1) % 1000 == 0:
                logging.info(f"  Compressed {i + 1:,} / {len(image_files):,} images")
    
    # Check zip file
    if os.path.exists(ZIP_FILE):
        zip_size = os.path.getsize(ZIP_FILE) / (1024**3)  # Size in GB
        logging.info(f"\n Zip archive created successfully!")
        logging.info(f"  • Filename: {ZIP_FILE}")
        logging.info(f"  • Size: {zip_size:.2f} GB")
        logging.info(f"  • Images: {len(image_files):,}")
        return True
    else:
        logging.error("Failed to create zip file!")
        return False

def main():
    """Main execution function."""
    print("IMAGE DOWNLOAD SCRIPT".center(60))
    print("Multi-Modal Food Discovery System".center(60))
    print()
    
    start_time = time.time()
    
    try:
        # Step 1: Create directories
        create_directories()
        
        # Step 2: Load dataset
        df = load_dataset()
        
        if len(df) == 0:
            logging.warning("No products with image URLs found. Exiting.")
            return
        
        # Step 3: Download images
        print(f"\n Ready to download images for {len(df):,} products")
        print("  Note: This may take several hours depending on your internet speed")
        print("    The script will resume from where it left off if interrupted")
        
        confirm = input("\nProceed with download? (y/n): ")
        if confirm.lower() != 'y':
            print("Download cancelled.")
            return
        
        stats = download_all_images(df)
        
        # Step 4: Create zip archive
        if stats['success'] + stats['exists'] > 0:
            if input("\n Create zip archive? (y/n): ").lower() == 'y':
                create_zip_archive()
        
        # Calculate total time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logging.info(f"\n Total time: {hours}h {minutes}m {seconds}s")
        logging.info("\n" + "="*60)
        logging.info(" SCRIPT COMPLETE! ".center(60))
        logging.info("="*60)
        
        if stats['success'] + stats['exists'] > 0:
            logging.info("\n Next steps:")
            logging.info("  1. Upload 'project_data_clean.parquet' to shared drive")
            logging.info("  2. Upload 'images.zip' to shared drive (if created)")
            logging.info("  3. Share the DATA_DRIVE_URL with the team")
            logging.info("  4. Proceed with Notebook 02 for baseline models")
        
    except KeyboardInterrupt:
        logging.warning("\n\n Script interrupted by user")
        logging.info("The script can be rerun and will skip already downloaded images")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
