import json
import networkx as nx
from pyvis.network import Network

print("1. Loading Graph Map...")
with open('crawler_state.json', 'r') as f:
    state = json.load(f)
    graph_data = state.get('graph_data', {})

print("2. Building NetworkX Graph & Calculating Importance...")
G = nx.DiGraph()
for source_page, target_links in graph_data.items():
    for target in target_links:
        G.add_edge(source_page, target)

# Calculate PageRank to figure out which nodes are most important
pagerank_scores = nx.pagerank(G)

# --- THE BROWSER SAVER ---
# Grab only the top 500 most important nodes so the browser doesn't crash
MAX_NODES = 500
top_nodes = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:MAX_NODES]
subgraph = G.subgraph(top_nodes)

print(f"3. Generating Interactive 3D Map for the top {MAX_NODES} pages...")

# Initialize a dark-mode, fullscreen interactive network
net = Network(height='100vh', width='100%', bgcolor='#1a1a1a', font_color='white', directed=True)
# Use Barnes-Hut physics (best for organic looking web graphs)
net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

# Add the nodes
for node in subgraph.nodes():
    score = pagerank_scores.get(node, 0)
    
    # Scale the visual size of the bubble based on PageRank
    # (Multiply by a large number because raw PageRank scores are tiny, e.g., 0.01)
    bubble_size = max(10, score * 3000) 
    
    # Create a clean label (e.g. 's1_1_2.html' instead of the massive URL)
    clean_label = node.split('/')[-1] if not node.endswith('/') else node.split('/')[-2]
    
    net.add_node(
        node, 
        label=clean_label, 
        title=node, # Shows full URL when you hover over it
        size=bubble_size,
        color='#4CAF50' # Nice matrix green
    )

# Add the connections (Edges)
for source, target in subgraph.edges():
    net.add_edge(source, target, color='#333333')

print("4. Saving to site_map.html...")
# Generate the HTML file
net.write_html('site_map.html')
print("--- Success! Open 'site_map.html' in Chrome or Edge! ---")