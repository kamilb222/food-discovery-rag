## 🚀 Project: Multi-Modal Food Discovery System

This is the central repository for our 15-week machine learning project. The goal is to build a multi-modal search and analysis system for food products, powered by the Open Food Facts dataset. This project uses large data files that are NOT stored in Git. To run this project, you must set up your local environment and download the data.

1.  **Clone the Repository:**
    `git clone ...`

2. Create a Virtual Environment

    `python -m venv venv`
    `source venv/bin/activate  # On Windows, use `venv\Scripts\activate``

2.  **Set Up Environment:**
    `pip install -r requirements.txt`

3.  **Set Up Private Links:**
    * This project uses a `.env` file to manage private links to our data.
    * Find the **`.env.example`** file in this repository.
    * Create a **new file** in the same directory named **`.env`** (just `.env`).
    * Copy the contents of `.env.example` into your new `.env` file.
    * Ask the team maintainer (e.g., via Discord/Slack) for the `DATA_DRIVE_URL` and paste it into your `.env` file.

4.  **Download the Data:**
    * Once your `.env` is set up, you can run the data download script (or manually go to the `DATA_DRIVE_URL`).
    * Download `project_data_clean.parquet` and `images.zip`.
    * Place them in the correct folders as specified by the project notebooks.

## Acknowledgements & License
This project is made possible by the Open Food Facts community.

The database contents are made available under the Open Database License (ODbL).

The individual contents of the database are available under the Database Contents License (DbCL).

The product images are available under the Creative Commons Attribution-ShareAlike 3.0 license.