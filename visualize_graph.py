from agent.graph import graph

# Generate Mermaid PNG
png_data = graph.get_graph().draw_mermaid_png()

# Save image
with open("airport_graph.png", "wb") as f:
    f.write(png_data)

print("Graph image saved as airport_graph.png")