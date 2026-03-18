import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin, urlparse, quote, urlsplit, urlunsplit
import time
import config
import json
import os

# --- Configuration Variables ---
BASE_URL = config.BASE_URL
MAX_PAGES = config.MAX_PAGES
CRAWL_DELAY = config.CRAWL_DELAY
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT
USER_AGENT = config.USER_AGENT

ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
IGNORED_PATTERNS = config.IGNORED_PATTERNS

CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP

COLLECTION_NAME = config.COLLECTION_NAME
MODEL_NAME = config.MODEL_NAME
EMBEDDING_DEVICE = config.EMBEDDING_DEVICE
DB_DIR = str(config.DB_DIR)  # Cast to string since it's a Path object
STATE_FILE = config.STATE_FILE

# --- BOUNDARY & STATE SETUP ---
parsed_base = urlparse(BASE_URL)
DOMAIN = parsed_base.netloc
# Extract the directory path to lock the crawler inside this specific folder
ALLOWED_PATH_PREFIX = (
    parsed_base.path.rsplit("/", 1)[0] + "/"
)  # e.g. "/matwis/amat/iss/"

print(f"Loading AI Model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)

print("Connecting to ChromaDB... ")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# --- LOAD PREVIOUS STATE ---
if os.path.exists(STATE_FILE):
    print(f"Loading previous crawler state from {STATE_FILE}...")
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
        visited_urls = set(state.get("visited_urls", []))
        urls_to_visit = state.get("urls_to_visit", [BASE_URL])
        graph_data = state.get("graph_data", {})
else:
    print("No previous state found. Starting fresh crawl...")
    visited_urls = set()
    urls_to_visit = [BASE_URL]
    graph_data = {}

# Use a Session for connection pooling
session = requests.Session()
# Set a User-Agent from config to avoid being blocked by some servers
session.headers.update({"User-Agent": USER_AGENT})


def is_valid_link(full_url):
    """Checks if a URL is safe to crawl."""
    parsed_url = urlsplit(full_url)
    # 1. Did it leave the university domain?
    if parsed_url.netloc != DOMAIN:
        return False
    # 2. Did it leak outside the target folder?
    if not parsed_url.path.startswith(ALLOWED_PATH_PREFIX):
        return False
    # 3. Does it have an extension, and is it allowed?
    ext = os.path.splitext(parsed_url.path)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        return False
    # 4. Is it a frame trap?
    for pattern in IGNORED_PATTERNS:
        if pattern in parsed_url.path:
            return False

    return True


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks based on word count."""
    words = text.split()
    chunks = []

    # Step through the words, moving forward by (chunk_size - overlap)
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

        # Stop if we've reached the end of the text
        if i + chunk_size >= len(words):
            break

    return chunks


def extract_page_data(url):
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f" [!] Failed to fetch {url}: {e}")
        return None, []

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    text = " ".join(soup.stripped_strings)

    new_links = []

    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(url, a_tag["href"])
        if urlsplit(full_url).netloc == DOMAIN:
            # 1. Strip the anchor tags
            no_anchor_url = full_url.split("#")[0]

            # --- Chrome-style Auto-Encoding ---
            parsed = urlsplit(no_anchor_url)
            safe_path = quote(parsed.path, safe="/;%")

            # Put the URL back together (urlunsplit takes exactly 5 arguments)
            clean_url = urlunsplit(
                (parsed.scheme, parsed.netloc, safe_path, parsed.query, "")
            )

            if is_valid_link(clean_url):
                new_links.append(clean_url)

    return text, new_links


print(f"\n--- Starting Spelunking Run on {BASE_URL} ---\n")
print(f"Locked to domain: {DOMAIN}")
print(f"Locked to path: {ALLOWED_PATH_PREFIX}\n")

pages_crawled = 0

start_time = time.time()  # 1. Record the start time here

try:
    while urls_to_visit and pages_crawled < MAX_PAGES:
        current_url = urls_to_visit.pop(0)

        if not is_valid_link(current_url):
            continue

        if current_url in visited_urls:
            continue

        print(f"[{pages_crawled + 1}] Crawling: {current_url}")
        text, new_links = extract_page_data(current_url)
        visited_urls.add(current_url)

        # Save graph connections
        graph_data[current_url] = new_links

        if text and len(text) > 50:
            # 1. Break the massive page into configurable word chunks
            text_chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for index, chunk in enumerate(text_chunks):
                chunk_id = f"{current_url}#chunk{index}"

                # 2. Add the E5 "passage:" prefix
                embedding = model.encode([f"passage: {chunk}"]).tolist()
                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=embedding,
                    metadatas=[{"url": current_url, "chunk_index": index}],
                )
            print(f"  -> Split and saved into {len(text_chunks)} overlapping chunks.")
        else:
            print("  -> Skipped (Not enough text content)")

        for link in new_links:
            if link not in visited_urls and link not in urls_to_visit:
                urls_to_visit.append(link)

        pages_crawled += 1
        time.sleep(CRAWL_DELAY)

    print(f"\n--- Crawl Complete! Total pages indexed this run: {pages_crawled} ---")

except KeyboardInterrupt:
    print("\n[!] Force quit detected. Saving progress before exiting...")

finally:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
    # --- SAVE STATE BEFORE EXITING ---
    # This block runs no matter what (success, error, or Ctrl+C)
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "visited_urls": list(visited_urls),
                "urls_to_visit": urls_to_visit,
                "graph_data": graph_data,
            },
            f,
        )
    print(
        f"State saved to {STATE_FILE}. Queue size: {len(urls_to_visit)}. You can resume the crawl later!"
    )
