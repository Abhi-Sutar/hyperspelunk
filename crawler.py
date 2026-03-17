import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin, urlparse
import time
import config
import json
import os

# --- BOUNDARY & STATE SETUP ---
parsed_base = urlparse(config.BASE_URL)
DOMAIN = parsed_base.netloc
# Extract the directory path to lock the crawler inside this specific folder 
ALLOWED_PATH_PREFIX = parsed_base.path.rsplit('/', 1)[0] + '/'  # e.g. "/matwis/amat/iss/"
# Files we DO NOT want to download
# IGNORED_EXTENSIONS = ('.pdf', '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.mp3', '.ogg', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')
# Allowed files
ALLOWED_EXTENSIONS = ('.html', '.htm', '.txt', '.md', '.php', '.asp', '.aspx')
# URLS containing these words will be skipped
IGNORED_PATTERNS = ['index.html']
STATE_FILE = "crawler_state.json"


print(f"Loading AI Model: {config.MODEL_NAME}...")
model = SentenceTransformer(config.MODEL_NAME, device="cuda")

print("Connecting to ChromaDB... ")
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_or_create_collection(name=config.COLLECTION_NAME)

# --- LOAD PREVIOUS STATE ---
if os.path.exists(STATE_FILE):
    print(f"Loading previous crawler state from {STATE_FILE}...")
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
        visited_urls = set(state.get('visited_urls', []))
        urls_to_visit = state.get('urls_to_visit', [config.BASE_URL])
        graph_data = state.get('graph_data', {})
else:
    print("No previous state found. Starting fresh crawl...")
    visited_urls = set()
    urls_to_visit = [config.BASE_URL]
    graph_data = {}

# Use a Session for connection pooling
session = requests.Session()
# Optional: Set a User-Agent to avoid being blocked by some servers
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

def is_valid_link(full_url):
    """Checks if a URL is safe to crawl."""
    parsed_url = urlparse(full_url)
    # 1. Did it leave the university domain?
    if parsed_url.netloc != DOMAIN:
        return False
    # 2. Did it leak outside the target folder?
    if not parsed_url.path.startswith(ALLOWED_PATH_PREFIX):
        return False
    # 3. Does it have an extension, and is it allowed?
    # os.path.splitext grabs the extension (e.g., '.html'). If there is no extension, it returns an empty string ''.
    ext = os.path.splitext(parsed_url.path)[1].lower()    
    # If an extension exists, BUT it's not in our allowed list, skip it!
    if ext and ext not in ALLOWED_EXTENSIONS:
        return False
    # 4. Is it a frame trap?
    for pattern in IGNORED_PATTERNS:
        if pattern in parsed_url.path:
            return False
    
    return True

def chunk_text(text, chunk_size=300, overlap=50):
    """Splits text into overlapping chunks based on word count."""
    words = text.split()
    chunks = []
    
    # Step through the words, moving forward by (chunk_size - overlap)
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # Stop if we've reached the end of the text
        if i + chunk_size >= len(words):
            break
            
    return chunks

def extract_page_data(url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f" [!] Failed to fetch {url}: {e}")
        return None, []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()

    text = ' '.join(soup.stripped_strings)

    new_links = []

    for a_tag in soup.find_all('a', href=True):
        full_url = urljoin(url, a_tag['href'])
        if urlparse(full_url).netloc == DOMAIN:
            clean_url = full_url.split('#')[0]

            if is_valid_link(clean_url):
                new_links.append(clean_url)

    return text, new_links

print(f"\n--- Starting Spelunking Run on {config.BASE_URL} ---\n")
print(f"Locked to domain: {DOMAIN}")
print(f"Locked to path: {ALLOWED_PATH_PREFIX}\n")

pages_crawled = 0

start_time = time.time()  # 1. Record the start time here

try:
    while urls_to_visit and pages_crawled < config.MAX_PAGES:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        print(f"[{pages_crawled + 1}] Crawling: {current_url}")
        text, new_links = extract_page_data(current_url)
        visited_urls.add(current_url)

        # Save graph connections
        graph_data[current_url] = new_links

        if text and len(text) > 50:
            # 1. Break the massive page into 300-word chunks
            text_chunks = chunk_text(text, chunk_size=300, overlap=50)
            for index, chunk in enumerate(text_chunks):
                chunk_id = f"{current_url}#chunk{index}"

                # 2. Add the E5 "passage:" prefix
                embedding = model.encode([f"passage: {chunk}"]).tolist()
                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=embedding,
                    metadatas=[{"url": current_url, "chunk_index": index}]
                )
            print(f"  -> Split and saved into {len(text_chunks)} overlapping chunks.")
        else:
            print("  -> Skipped (Not enough text content)")

        for link in new_links:
            if link not in visited_urls and link not in urls_to_visit:
                urls_to_visit.append(link)

        pages_crawled += 1
        time.sleep(config.CRAWL_DELAY)

    print(f"\n--- Crawl Complete! Total pages indexed this run: {pages_crawled} ---")

except KeyboardInterrupt:
     print("\n[!] Force quit detected. Saving progress before exiting...")

finally:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
    # --- SAVE STATE BEFORE EXITING ---
    # This block runs no matter what (success, error, or Ctrl+C)
    with open(STATE_FILE, 'w') as f:
        json.dump({
            'visited_urls': list(visited_urls),
            'urls_to_visit': urls_to_visit,
            'graph_data': graph_data
        }, f)
    print(f"State saved to {STATE_FILE}. Queue size: {len(urls_to_visit)}. You can resume the crawl later!")
