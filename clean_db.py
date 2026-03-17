import chromadb
import config

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_collection(name=config.COLLECTION_NAME)

# Fetch everything so we can hunt for bad extensions
print("Scanning database for zombie files...")
all_data = collection.get(include=["metadatas"])

bad_ids = []
for idx, metadata in enumerate(all_data["metadatas"]):
    url = metadata["url"].lower()
    # If the URL ends with .pdf, .doc, or .zip, mark it for deletion
    if url.endswith(".pdf") or url.endswith(".doc") or url.endswith(".zip"):
        bad_ids.append(all_data["ids"][idx])
        print(f"Found bad file: {url} (ID: {all_data['ids'][idx]})")

if bad_ids:
    print(f"Found {len(bad_ids)} infected chunks. Vaporizing...")
    # Delete them all in one shot
    collection.delete(ids=bad_ids)
    print("Database cleansed!")
else:
    print("Database is clean! No bad files found.")
