# hyperspelunk

`hyperspelunk` deep-crawls a specific website, builds a local multilingual semantic-search
index over its pages, and layers classic web-graph analysis (PageRank, HITS hub/authority)
on top so search results are ranked by both textual relevance and structural importance in
the site's link graph. It also produces exploratory visualizations of the crawled site
(a Gephi export and an interactive HTML site map).

The default target checked into the (git-ignored) local config is a university "Introduction
to Solid State Physics" course site, but `hyperspelunk` works against any single site you
point it at.

## How it works

The pipeline runs as a sequence of standalone scripts, all driven by a single `config.py`
file. There are no CLI entry points — every script is run directly with `uv run python
<script>.py` and reads its behavior from `config.py`.

1. **`crawler.py` — crawl + index.** A breadth-first crawler starting at `config.BASE_URL`,
   restricted to the same domain and to the base URL's path prefix. For each page it strips
   boilerplate HTML (`script`/`style`/`nav`/`footer`/`header`) with BeautifulSoup, splits the
   remaining text into overlapping word-count chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP`), embeds
   each chunk with a multilingual sentence-transformer model, and upserts the embeddings into
   a ChromaDB collection. Outgoing links from each page are recorded into a link graph. Crawl
   progress (visited URLs, the BFS queue, and the link graph) is persisted to
   `crawler_state.json` after every page and on `Ctrl+C`, so a crawl can be safely stopped and
   resumed.

2. **`clean_db.py` — optional cleanup.** Scans the ChromaDB collection and deletes any chunk
   whose source URL ends in `.pdf`, `.doc`, or `.zip` — useful if non-HTML documents slipped
   past the crawler's extension filtering.

3. **`inject_metrics.py` — graph metrics.** Builds a directed graph from
   `crawler_state.json`'s link data and computes **PageRank** (overall page importance from
   inbound links) and **HITS** hub/authority scores (a "hub" links out to many good pages; an
   "authority" is linked to by many good hubs) using `networkx`. These scores are written back
   into every chunk's metadata in ChromaDB, in batches. (`inject_pagerank.py` is an older,
   PageRank-only predecessor kept for reference — prefer `inject_metrics.py`.)

4. **`search.py` — search.** An interactive command-line search loop. Each query is embedded
   and matched against ChromaDB to fetch the top 15 nearest chunks by semantic distance, which
   are then **re-ranked** using the PageRank/HITS metadata injected in step 3:
   - Pages get a capped "boost" from their PageRank and HITS authority scores, which lowers
     (improves) their effective distance.
   - Pages that are strong link hubs but weak authorities are penalized, since hub-like pages
     (navigation/index pages) tend to be low on actual content.
   - This boost only ever applies if the raw semantic distance is already below a relevance
     threshold — so a structurally important but semantically irrelevant page is never
     rescued by its link-graph reputation.

   Results are deduplicated by URL (best-scoring chunk per page) down to a handful of top
   matches, printed with their score breakdown and a text snippet.

5. **`export_gephi.py` / `visualize.py` — optional visualization.** `export_gephi.py` writes
   the crawl's link graph plus its PageRank/HITS scores to `course_universe.graphml` for
   import into [Gephi](https://gephi.org/). `visualize.py` builds a self-contained, pre-laid-
   out interactive HTML graph (`site_map.html`) of the top pages by PageRank.

6. **`onnx_export.py` / `onnx_search.py` — experimental alternate search path.** Exports a
   sentence-transformer model to ONNX and runs query embedding via ONNX Runtime directly
   instead of `sentence-transformers`, for faster inference. This path does not implement the
   PageRank/HITS re-ranking from `search.py` — it's a performance experiment, not the primary
   workflow.

7. **`test_search.py` — smoke test.** Seeds a few hardcoded multilingual dummy documents into
   ChromaDB and runs a sample query, to sanity-check the embedding + ChromaDB pipeline without
   needing a real crawl first.

### Data flow

```
config.py ─┬─▶ crawler.py ──▶ ChromaDB (my_vector_db/) + crawler_state.json
           │                         │                          │
           │                  clean_db.py (optional)     export_gephi.py ──▶ course_universe.graphml
           │                         │                          │
           │                  inject_metrics.py           visualize.py ──▶ site_map.html
           │                         │
           └────────────────▶  search.py  (interactive query loop)
```

## Setup

**Requirements:** Python 3.11 (see `.python-version`) and [uv](https://docs.astral.sh/uv/).

1. Install dependencies. `torch` is pulled from a hardware-specific package index, so choose
   one of:
   ```bash
   uv sync --extra cpu          # CPU-only
   uv sync --extra gpu-cu126    # NVIDIA GPU, CUDA 12.6
   ```

2. Create your local config. `config.py` is git-ignored (it's user- and site-specific), so
   copy the tracked template and edit it:
   ```bash
   cp example_config.py config.py
   ```
   At minimum, set `BASE_URL` to the site you want to crawl. Other settings you may want to
   adjust, grouped as they appear in `config.py`:
   - **Crawler settings** — `MAX_PAGES` (crawl budget), `CRAWL_DELAY` (politeness delay
     between requests), `REQUEST_TIMEOUT`, `USER_AGENT`.
   - **URL filtering** — `ALLOWED_EXTENSIONS`, `IGNORED_PATTERNS`.
   - **Text processing** — `CHUNK_SIZE` / `CHUNK_OVERLAP` (word-count chunking window).
   - **Vector database** — `COLLECTION_NAME`, `MODEL_NAME` (the sentence-transformer model),
     `EMBEDDING_DEVICE`.
   - **Search settings** — `FETCH_LIMIT`, `TOP_UNIQUE_RESULTS`, `DISTANCE_THRESHOLD`,
     `PR_MULTIPLIER`, `AUTH_MULTIPLIER`, `HUB_PENALTY`, `MAX_BOOST` (see "Known limitations"
     below — these currently aren't read by `search.py`).
   - **Graph metrics** — `METRICS_BATCH_SIZE`.

   If you change `MODEL_NAME` to a non-E5 model, note that `crawler.py` and `search.py` prefix
   text with `"passage: "` / `"query: "` respectively, which is an E5-model convention — remove
   those prefixes if you switch to a model that doesn't expect them.

## Usage

Run the pipeline in order:

```bash
uv run python crawler.py         # crawl the site, build the vector index + link graph
uv run python clean_db.py        # optional: purge stray pdf/doc/zip chunks
uv run python inject_metrics.py  # compute PageRank + HITS, required for search re-ranking
uv run python search.py          # interactive semantic search
```

`crawler.py` can be interrupted with `Ctrl+C` at any time and re-run later to resume from
where it left off. `search.py` starts an interactive prompt — type a query, or `quit`/`exit`/
`q` to leave.

Optional exploration/visualization, once you have a crawl and metrics:

```bash
uv run python export_gephi.py    # writes course_universe.graphml, for Gephi
uv run python visualize.py       # writes site_map.html, a standalone interactive site map
```

Sanity-check the embedding/search pipeline without crawling anything:

```bash
uv run python test_search.py
```

Experimental ONNX-based search path (separate from the ChromaDB re-ranking above):

```bash
uv run python onnx_export.py     # exports a model to ./onnx_model/
uv run python onnx_search.py     # search using ONNX Runtime instead of sentence-transformers
```

## Known limitations

- There are no CLI arguments anywhere in the pipeline — everything is controlled through
  `config.py`, and every script is invoked as `uv run python <script>.py`.
- `search.py`'s re-ranking constants (`PR_MULTIPLIER`, `AUTH_MULTIPLIER`, `HUB_PENALTY`,
  `MAX_BOOST`, the `0.42` distance threshold) are currently hardcoded inside the script rather
  than read from `config.py`, even though matching settings exist there under "Search
  settings" — editing those values in `config.py` has no effect today.
- `crawler.py` currently hardcodes the embedding model to `device="cuda"` and will fail
  without a CUDA GPU. `search.py`, by contrast, auto-detects CUDA vs. CPU and falls back
  gracefully.
- `requirement.txt` and `main.py` are unrelated leftovers (a stray `pip freeze` dump and `uv
  init` boilerplate) — not part of the actual pipeline.
