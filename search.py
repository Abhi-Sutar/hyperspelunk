import chromadb
from sentence_transformers import SentenceTransformer
import config
import textwrap
import os

# Silence Hugging Face token warnings for a cleaner terminal
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print(f"Loading AI Model: {config.MODEL_NAME} onto GPU...")
model = SentenceTransformer(config.MODEL_NAME, device="cuda")

print("Connecting to ChromaDB... ")
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_or_create_collection(name=config.COLLECTION_NAME)

# Check how many chunks we have
chunk_count = collection.count()
if chunk_count == 0:
    print("[!] Your database is empty. Run crawler.py first!")
    exit()
# Get all metadata to count unique pages
all_metadata = collection.get(include=["metadatas"])
unique_pages = set(meta['url'] for meta in all_metadata['metadatas'])
page_count = len(unique_pages)

print(f"\n--- Spelunk Search Ready! ({chunk_count} chunks across {page_count} pages indexed) ---")

def search_index(query, top_unique=3, fetch_limit=15):
    """Embeds the query, fetches chunks, applies PageRank boost, and groups unique URLs."""
    query_embedding = model.encode([f"query: {query}"]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_limit
    )

    # Check if we got results
    if not results['documents'][0]:
        print("\nNo relevant matches found.")
        return
    
    # 1. Fetch all raw results and calculate the Adjusted Score
    raw_results = []

    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        url = results['metadatas'][0][i]['url']
        distance = results['distances'][0][i]
        # Get the PageRank score from metadata (default to 0 if not present)
        pagerank = results['metadatas'][0][i].get('pagerank', 0.0)
        # Get the HITS scores from metadata (default to 0 if not present)
        authority = results['metadatas'][0][i].get('authority', 0.0)

        # --- RE-RANKING MATH (WITH CEILING) ---
        # Distance: lower is better. PageRank: higher is better.
        # We artificially lower the distance score if the page is authoritative.
        PR_MULTIPLIER = 3.0   
        AUTH_MULTIPLIER = 5.0 # We trust Authority (HITS) more than raw PageRank
        MAX_ALLOWED_BOOST = 0.08 # The absolute maximum we will artificially lower the distance

        # --- THE CONTEXT GATE ---
        # If the semantic distance is worse than 0.42, it's a bad text match.
        # Do NOT let Authority/PageRank rescue it. Give it zero boost.
        if distance > 0.42:
            applied_boost = 0.0
        else:
            raw_boost = (pagerank * PR_MULTIPLIER) + (authority * AUTH_MULTIPLIER)
            applied_boost = min(raw_boost, MAX_ALLOWED_BOOST)  # Cap the boost to prevent over-boosting
            adjusted_score = distance - applied_boost

        raw_results.append({
            "url": url,
            "text": text,
            "distance": distance,
            "pagerank": pagerank,
            "authority": authority,
            "applied_boost": applied_boost,
            "adjusted_score": adjusted_score
        })

    # 2. Sort the results by the new adjusted score (lowest is best)
    raw_results.sort(key=lambda x: x['adjusted_score'])


    print(f"\n--- Top {top_unique} Results for '{query}' ---\n")

    seen_urls = set()
    matches_found = 0

    for match in raw_results:
        text = match['text']
        url = match['url']
        distance = match['distance']
        pagerank = match['pagerank']
        adjusted_score = match['adjusted_score']

        # # E5 Quality Threshold (0.45 is a solid cutoff for "good" matches)
        # if score > 0.45: 
        #     continue

        # DEDUPLICATION: If we already showed this URL, skip to the next one
        if url in seen_urls:
            continue

        # We found a new unique page!
        seen_urls.add(url)
        matches_found += 1

        # Clean up the text for the terminal (truncate to 250 chars)
        snippet = textwrap.shorten(text, width=250, placeholder=" ... [read more]")

        print(f"Match {matches_found} (Score: {adjusted_score:.4f})")
        print(f"Link: {url}")
        print(f"PageRank: {pagerank:.4f}  | Authority: {authority:.4f} | Semantic Distance: {distance:.4f}")
        print(f"BOOST APPLIED: {match['applied_boost']:.4f} (Raw Boost: {match['pagerank'] * PR_MULTIPLIER + match['authority'] * AUTH_MULTIPLIER:.4f})")
        print(f"Text: {snippet}\n")
        print("-" * 60 + "\n")

        # Stop once we hit our target number of unique pages
        if matches_found >= top_unique:
            break

    if matches_found == 0:
        print("No highly relevant matches found (results were below the quality threshold).")

# --- The Interactive Loop ---
while True:
    try:
        user_query = input("\nEnter search term (or 'quit' to exit): ").strip()

        if not user_query:
            continue
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Exiting Spelunk Search...")
            break

        search_index(user_query)

    except KeyboardInterrupt:
        # Handles Ctrl+C gracefully
        print("\nExiting Spelunk Search...")
        break