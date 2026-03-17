import json
import chromadb
import networkx as nx
import config

print("1. Loading Graph Map...")
with open("crawler_state.json", "r") as f:
    state = json.load(f)
    graph_data = state.get("graph_data", {})


print("2. Calculating Global PageRank...")
G = nx.DiGraph()
for source_page, target_links in graph_data.items():
    for target in target_links:
        G.add_edge(source_page, target)

# This calculates the authority of every page (scores usually range from 0.0001 to 0.05)
pagerank_scores = nx.pagerank(G)

print("3. Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_collection(name=config.COLLECTION_NAME)


print("4. Injecting PageRank into Vector Metadata...")
# Pull ALL items from the database so we can update them
all_data = collection.get(include=["metadatas"])

update_ids = []
update_metadatas = []

for i in range(len(all_data["ids"])):
    chunk_id = all_data["ids"][i]
    metadata = all_data["metadatas"][i]
    url = metadata["url"]

    # Get the rank for this URL (default to 0 if it's an orphan page)
    rank = pagerank_scores.get(url, 0.0)

    # Add the rank to the chunk's metadata
    metadata["pagerank"] = rank

    update_ids.append(chunk_id)
    update_metadatas.append(metadata)

# Push the updated metadata back into ChromaDB in one massive batch
collection.update(ids=update_ids, metadatas=update_metadatas)

print("--- Injection Complete! Your Vector DB is now Graph-Aware! ---")

# Sort and print the top 10 most important pages
top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

print("--- The 10 Most Important Pages on the Site ---")
for url, score in top_pages[:10]:
    print(f"Score: {score:.4f} | {url}")
