# hyperspelunk

hyperspelunk deep-crawls a specific website, builds a local multilingual semantic-search index over its pages, and layers classic web-graph analysis (PageRank and HITS) on top so search results are ranked by both textual relevance and structural importance in the site link graph.

It also produces exploratory visualizations of the crawl (a Gephi export and an interactive HTML site map).

The default target checked into local config is a university Introduction to Solid State Physics course site, but hyperspelunk works against any single site you point it at.

## Quick Start

```bash
uv sync --extra cpu
cp example_config.py config.py
uv run python crawler.py
uv run python inject_metrics.py
uv run python search.py
```

---

<details>
<summary><strong>How It Works</strong></summary>

<br>

The pipeline runs as a sequence of standalone scripts, all driven by config.py.
There are no CLI entry points. Every script is run directly with uv run python SCRIPT.py and reads behavior from config.py.

1. **crawler.py (crawl + index)**
   A breadth-first crawler starting at BASE_URL, restricted to the same domain and base path prefix.
   For each page, it strips boilerplate HTML (script/style/nav/footer/header), splits text into overlapping word-count chunks (CHUNK_SIZE and CHUNK_OVERLAP), embeds chunks with a multilingual sentence-transformer model, and upserts embeddings into ChromaDB.
   Outgoing links are recorded into a link graph. Crawl progress (visited URLs, queue, graph) is persisted to crawler_state.json after every page and on Ctrl+C, so crawling can resume safely.

2. **clean_db.py (optional cleanup)**
   Scans ChromaDB and deletes chunks whose source URL ends in .pdf, .doc, or .zip.

3. **inject_metrics.py (graph metrics)**
   Builds a directed graph from crawler_state.json link data and computes:
   - PageRank
   - HITS hub scores
   - HITS authority scores

   Scores are written back into ChromaDB chunk metadata in batches.
   inject_pagerank.py is an older PageRank-only predecessor kept for reference.

4. **search.py (interactive search)**
   Each query is embedded and matched against ChromaDB to fetch nearest chunks by semantic distance, then re-ranked using PageRank/HITS metadata:
   - Capped boost from PageRank and authority lowers effective distance.
   - Hub-strong but authority-weak pages are penalized.
   - Boost applies only below a semantic distance threshold, so semantically irrelevant pages are never rescued by graph reputation.

   Results are deduplicated by URL (best chunk per page), then top matches are shown with score breakdown and snippet.

5. **export_gephi.py and visualize.py (optional visualization)**
   - export_gephi.py writes course_universe.graphml for Gephi.
   - visualize.py writes site_map.html, a standalone interactive graph.

6. **onnx_export.py and onnx_search.py (experimental alternate path)**
   Exports a sentence-transformer model to ONNX and runs query embedding via ONNX Runtime for faster inference.
   This path does not include PageRank/HITS re-ranking from search.py.

7. **test_search.py (smoke test)**
   Seeds hardcoded multilingual dummy docs into ChromaDB and runs a sample query to sanity-check embedding and retrieval without crawling first.

</details>

<details>
<summary><strong>Data Flow</strong></summary>

<br>

```text
config.py
  -> crawler.py
     -> ChromaDB (my_vector_db/)
     -> crawler_state.json
        -> export_gephi.py -> course_universe.graphml
        -> visualize.py -> site_map.html
  -> clean_db.py (optional)
  -> inject_metrics.py
  -> search.py (interactive query loop)
```

</details>

<details open>
<summary><strong>Setup</strong></summary>

<br>

Requirements:
- Python 3.11 (see .python-version)
- uv: https://docs.astral.sh/uv/

1. Install dependencies. Torch is pulled from a hardware-specific package index, so choose one:
```bash
uv sync --extra cpu
uv sync --extra gpu-cu126
```

2. Create local config:
```bash
cp example_config.py config.py
```

At minimum, set BASE_URL to the site you want to crawl.

Settings you may want to adjust in config.py:
- Crawler settings: MAX_PAGES, CRAWL_DELAY, REQUEST_TIMEOUT, USER_AGENT
- URL filtering: ALLOWED_EXTENSIONS, IGNORED_PATTERNS
- Text processing: CHUNK_SIZE, CHUNK_OVERLAP
- Vector database: COLLECTION_NAME, MODEL_NAME, EMBEDDING_DEVICE
- Search settings: FETCH_LIMIT, TOP_UNIQUE_RESULTS, DISTANCE_THRESHOLD, PR_MULTIPLIER, AUTH_MULTIPLIER, HUB_PENALTY, MAX_BOOST
- Graph metrics: METRICS_BATCH_SIZE

If you change MODEL_NAME to a non-E5 model, note that crawler.py and search.py currently prefix text with passage: and query: respectively. Remove those prefixes for models that do not expect them.

</details>

<details open>
<summary><strong>Usage</strong></summary>

<br>

Run the main pipeline in order:

```bash
uv run python crawler.py
uv run python clean_db.py
uv run python inject_metrics.py
uv run python search.py
```

Notes:
- crawler.py can be interrupted with Ctrl+C and resumed later.
- search.py starts an interactive prompt. Type quit, exit, or q to leave.

Optional exploration and visualization:

```bash
uv run python export_gephi.py
uv run python visualize.py
```

Sanity-check the embedding/search pipeline without crawling:

```bash
uv run python test_search.py
```

Experimental ONNX search path:

```bash
uv run python onnx_export.py
uv run python onnx_search.py
```

</details>

<details>
<summary><strong>Known Limitations</strong></summary>

<br>

- No CLI arguments in the current pipeline. Everything is controlled through config.py.
- search.py re-ranking constants (PR_MULTIPLIER, AUTH_MULTIPLIER, HUB_PENALTY, MAX_BOOST, and the 0.42 threshold) are hardcoded today instead of reading matching values from config.py.
- crawler.py currently hardcodes embedding device as CUDA and fails without a CUDA GPU; search.py auto-detects CUDA vs CPU.
- requirement.txt and main.py are unrelated leftovers and not part of the actual pipeline.

</details>
