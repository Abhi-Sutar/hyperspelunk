# config.py

# --- CRAWLER SETTINGS ---
# Replace with the actual target URL
BASE_URL = "http://the-obscure-link.com"
MAX_PAGES = 50  # Safety limit for your first run
CRAWL_DELAY = 1  # Seconds to wait between page loads (Be polite!)

# --- AI & DATABASE SETTINGS ---
DB_DIR = "./my_vector_db"
COLLECTION_NAME = "obscure_pages"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
