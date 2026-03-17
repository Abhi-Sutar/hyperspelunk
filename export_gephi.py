import json
import networkx as nx

print("1. Loading Graph Map...")
with open("crawler_state.json", "r") as f:
    state = json.load(f)
    graph_data = state.get("graph_data", {})

print("2. Building Full NetworkX Graph (All 6,000+ nodes)...")
G = nx.DiGraph()
for source_page, target_links in graph_data.items():
    for target in target_links:
        G.add_edge(source_page, target)

print("3. Calculating Metrics for Gephi...")
# We calculate these here so Gephi can use them to automatically size and color nodes
pagerank_scores = nx.pagerank(G)
hubs, authorities = nx.hits(G, max_iter=1000)

print("4. Baking data into the nodes...")
for node in G.nodes():
    # Save the math scores directly inside the node
    G.nodes[node]["pagerank"] = pagerank_scores.get(node, 0.0)
    G.nodes[node]["authority"] = authorities.get(node, 0.0)
    G.nodes[node]["hub"] = hubs.get(node, 0.0)

    # Create a clean, readable label for Gephi to display
    clean_label = node.split("/")[-1] if not node.endswith("/") else node.split("/")[-2]
    G.nodes[node]["label"] = clean_label

print("5. Exporting to GraphML format...")
nx.write_graphml(G, "course_universe.graphml")
print("--- Success! 'course_universe.graphml' is ready for Gephi! ---")
