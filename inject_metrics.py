import json
import chromadb
import networkx as nx
import config

print("1. Loading Graph Map...")
with open('crawler_state.json', 'r') as f:
    state = json.load(f)
    graph_data = state.get('graph_data', {})

print("2. Calculating Global Graph Metrics (PageRank & HITS)...")
G = nx.DiGraph()
for source_page, target_links in graph_data.items():
    for target in target_links:
        G.add_edge(source_page, target)

# Calculate both metrics!
pagerank_scores = nx.pagerank(G)
hubs, authorities = nx.hits(G, max_iter=1000)

print("3. Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_collection(name=config.COLLECTION_NAME)

print("4. Injecting Scores into Vector Metadata...")
all_data = collection.get(include=["metadatas"])

update_ids = []
update_metadatas = []

for i in range(len(all_data['ids'])):
    chunk_id = all_data['ids'][i]
    metadata = all_data['metadatas'][i]
    url = metadata['url']
    
    # Get the scores for this URL (default to 0 if it's an orphan page)
    pagerank = pagerank_scores.get(url, 0.0)
    hub_score = hubs.get(url, 0.0)
    authority_score = authorities.get(url, 0.0)
    
    # Add the scores to the chunk's metadata
    metadata['pagerank'] = pagerank
    metadata['hub'] = hub_score
    metadata['authority'] = authority_score
    
    update_ids.append(chunk_id)
    update_metadatas.append(metadata)

# Push the updated metadata back into ChromaDB in safe batches
BATCH_SIZE = 5000
print(f"Pushing {len(update_ids)} updates to ChromaDB in batches of {BATCH_SIZE}...")

for i in range(0, len(update_ids), BATCH_SIZE):
    batch_ids = update_ids[i:i + BATCH_SIZE]
    batch_metadatas = update_metadatas[i:i + BATCH_SIZE]
    
    collection.update(ids=batch_ids, metadatas=batch_metadatas)
    print(f" -> Successfully injected batch ({i + len(batch_ids)} / {len(update_ids)})")

print("--- Injection Complete! Your DB has both PageRank AND Authority! ---")

print("--- The 10 Most Important Pages on the Site (by PageRank) ---")
for url, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Score: {score:.4f} | {url}")

print("\n--- The 10 Best Hubs on the Site (by HITS Hub Score) ---")
for url, score in sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Hub Score: {score:.4f} | {url}")

print("\n--- The 10 Most Authoritative Pages on the Site (by HITS Authority Score) ---")
for url, score in sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"Authority Score: {score:.4f} | {url}")