import chromadb
from sentence_transformers import SentenceTransformer
import config

print(f"Loading AI Model: {config.MODEL_NAME} onto RTX 3060...")
model = SentenceTransformer(config.MODEL_NAME, device="cuda")

print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_or_create_collection(name=config.COLLECTION_NAME)


# Dummy data: English, German, Latin
documents = [
    "The secret treasure is buried under the old oak tree.",
    "Der verborgene Schatz liegt unter der alten Eiche begraben.",
    "Thesaurus occultus sub vetere quercu sepultus est.",
]

metadatas = [{"url": "/en.html"}, {"url": "/de.html"}, {"url": "/la.html"}]
ids = ["test_doc1", "test_doc2", "test_doc3"]

print("Embedding documents...")
embeddings = model.encode(documents).tolist()

print("Saving documents to ChromaDB...")
collection.upsert(
    ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
)

# The Search
search_term = "hidden gold"
print(f"\n--- Searching database for: '{search_term}' ---")

print("[DIAGNOSTIC] 1. Asking AI to translate search term...")
query_embedding = model.encode([search_term]).tolist()

print("[DIAGNOSTIC] 2. Translation successful! Asking ChromaDB to search...")
results = collection.query(query_embeddings=query_embedding, n_results=3)

print(
    f"[DIAGNOSTIC] 3. Search successful! Found {len(results['documents'][0])} matches. Printing now..."
)

for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    url = results["metadatas"][0][i]["url"]
    score = results["distances"][0][i]
    print(f"\nMatch {i + 1} (Distance: {score:.4f}):\nURL: {url}\nText: {doc}")
