from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
STATE_FILE = BASE_DIR / "crawler_state.json"
DB_DIR = BASE_DIR / "my_vector_db"

# --- Crawler Settings ---
BASE_URL = "http://the-obscure-link.com"  # Replace with target
MAX_PAGES = 50
CRAWL_DELAY = 1.0  # seconds between requests
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# --- URL Filtering ---
ALLOWED_DOMAIN = None  # will be set from BASE_URL
ALLOWED_PATH_PREFIX = None  # will be set from BASE_URL
ALLOWED_EXTENSIONS = (".html", ".htm", ".txt", ".md", ".php", ".asp", ".aspx")
IGNORED_PATTERNS = ["index.html"]

# --- Text Processing ---
CHUNK_SIZE = 300  # words
CHUNK_OVERLAP = 50

# --- Vector Database ---
COLLECTION_NAME = "obscure_pages"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cuda"  # will be auto-detected if None

# --- Search Settings ---
FETCH_LIMIT = 15
TOP_UNIQUE_RESULTS = 3
DISTANCE_THRESHOLD = 0.42
PR_MULTIPLIER = 3.0
AUTH_MULTIPLIER = 5.0
HUB_PENALTY = 10.0
MAX_BOOST = 0.08

# --- Graph Metrics ---
METRICS_BATCH_SIZE = 5000
