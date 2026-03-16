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

# Check how many pages we actually have to search through
doc_count = collection.count()
print(f"\n--- Spelunk Search Ready! ({doc_count} pages indexed) ---")

if doc_count == 0:
    print("[!] Your database is empty. Run crawler.py first!")
    exit()

def search_index(query, top_k=3):
    """Embeds the query and searches the vector database."""
    query_embedding = model.encode([f"query: {query}"]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # Check if we got results
    if not results['documents'][0]:
        print("\nNo relevant matches found.")
        return

    print(f"\n--- Top {top_k} Results for '{query}' ---\n")
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        url = results['metadatas'][0][i]['url']
        score = results['distances'][0][i]

        # Clean up the text for the terminal (truncate to 250 chars)
        snippet = textwrap.shorten(text, width=250, placeholder=" ... [read more]")

        print(f"Match {i+1} (Distance: {score:.4f})")
        print(f"Link: {url}")
        print(f"Text: {snippet}\n")
        print("-" * 60 + "\n")

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