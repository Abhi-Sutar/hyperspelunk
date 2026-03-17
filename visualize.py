import json
import networkx as nx
from pyvis.network import Network

print("1. Loading Graph Map...")
with open("crawler_state.json", "r") as f:
    state = json.load(f)
    graph_data = state.get("graph_data", {})

print("2. Building NetworkX Graph & Calculating Importance...")
G = nx.DiGraph()
for source_page, target_links in graph_data.items():
    for target in target_links:
        G.add_edge(source_page, target)

pagerank_scores = nx.pagerank(G)

MAX_NODES = 500
top_nodes = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:MAX_NODES]
subgraph = G.subgraph(top_nodes)

print("3. Pre-calculating layout in Python (Offloading math from the browser)...")
# nx.spring_layout calculates the perfect "equilibrium" for the graph
# 'k' controls the optimal distance between nodes. 'iterations' ensures it settles completely.
pos = nx.spring_layout(subgraph, k=0.15, iterations=50)

print(f"4. Generating STATIC Interactive Map for the top {MAX_NODES} pages...")

net = Network(
    height="100vh", width="100%", bgcolor="#1a1a1a", font_color="white", directed=True
)

# --- THE FIX: Turn off the browser's physics engine entirely ---
net.toggle_physics(False)

for node in subgraph.nodes():
    score = pagerank_scores.get(node, 0)
    bubble_size = max(10, score * 3000)
    clean_label = node.split("/")[-1] if not node.endswith("/") else node.split("/")[-2]

    # Grab the pre-calculated X and Y coordinates and scale them up for the screen
    x_coord = float(pos[node][0] * 2000)
    y_coord = float(pos[node][1] * 2000)

    net.add_node(
        node,
        label=clean_label,
        title=node,
        size=bubble_size,
        color="#4CAF50",
        x=x_coord,  # Hardcode the X position
        y=y_coord,  # Hardcode the Y position
        physics=False,  # Explicitly forbid this node from moving
    )

for source, target in subgraph.edges():
    net.add_edge(source, target, color="#333333")

print("5. Saving to site_map.html...")
net.write_html("site_map.html")
print("--- Success! Open 'site_map.html' for instant, lag-free viewing! ---")
