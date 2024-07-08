# %%
print("hello world")
# %%
import einops


# %%
import torch

# Example tensor of shape (d_sae, d_sae, n_layers)
d_sae = 4  # example dimension
n_layers = 3  # example number of layers
tensor = torch.rand(d_sae, d_sae, n_layers)

# Reshape the tensor to (n_layers, d_sae, d_sae)
reshaped_tensor = einops.rearrange(tensor, 'd1 d2 n -> n d1 d2')

# Print the shapes to verify
print("Original shape:", tensor.shape)
print("Reshaped shape:", reshaped_tensor.shape)

# %%

import igraph as ig

# Create the igraph graph
g = ig.Graph(directed=True)

# Add nodes to the graph
for layer in range(n_layers):
    for node in range(d_sae):
        g.add_vertex(name=f'layer_{layer}_node_{node}')


# Add directed edges with weights between consecutive layers
for layer in range(n_layers - 1):
    for source_node in range(d_sae):
        for target_node in range(d_sae):
            source = g.vs.find(name=f'layer_{layer}_node_{source_node}')
            target = g.vs.find(name=f'layer_{layer + 1}_node_{target_node}')
            weight = tensor[source_node, target_node, layer]
            g.add_edge(source, target, weight=weight)

# Print summary of the graph
print(g.summary())


# %%
# Plot the graph using matplotlib
import matplotlib.pyplot as plt

layout = g.layout('kk')
visual_style = {}
visual_style["vertex_label"] = g.vs["name"]
visual_style["layout"] = layout

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    **visual_style
)
plt.savefig('graph_plot.png')

print("Graph plot saved as 'graph_plot.png'")
# %%
# Define the layout manually
layout = []
for layer in range(n_layers):
    for node in range(d_sae):
        layout.append((layer, -node))  # Position each layer in a column

# Plot the graph using matplotlib
visual_style = {}
visual_style["vertex_label"] = g.vs["name"]
visual_style["layout"] = layout

fig, ax = plt.subplots(figsize=(8, 6))
ig.plot(
    g,
    target=ax,
    layout=layout,
    vertex_label=g.vs["name"],
    bbox=(600, 400),
    margin=20
)
plt.savefig('graph_plot.png')

print("Graph plot saved as 'graph_plot.png'")
# %%
import leidenalg
# Step 2: Perform the Leiden algorithm
partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

# Step 3: Visualize the graph with detected communities
# Set up the colors for the communities
palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
g.vs['color'] = [palette[community] for community in partition.membership]

# Define the layout manually (left to right in layers)
layout = []
for layer in range(n_layers):
    for node in range(d_sae):
        layout.append((layer, -node))  # Position each layer in a column

# Plot the graph using matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ig.plot(
    g,
    target=ax,
    layout=layout,
    vertex_size=20,
    vertex_color=g.vs['color'],
    vertex_label=g.vs['name'],
    bbox=(600, 400),
    margin=20
)
plt.savefig('graph_plot.png', dpi=300)
plt.show()

print("Graph plot saved as 'graph_plot.png'")
# %%
# Example tensor of shape (d_sae, d_sae, n_layers)
d_sae = 4  # example dimension
n_layers = 3  # example number of layers
tensor = torch.rand(d_sae, d_sae, n_layers)

# Create the igraph graph
g = ig.Graph(directed=True)

# Add nodes to the graph
for layer in range(n_layers):
    for node in range(d_sae):
        g.add_vertex(name=f'layer_{layer}_node_{node}')

# Collect edges and weights from the tensor
edges = []
weights = []
for layer in range(n_layers - 1):
    for source_node in range(d_sae):
        for target_node in range(d_sae):
            source = g.vs.find(name=f'layer_{layer}_node_{source_node}').index
            target = g.vs.find(name=f'layer_{layer + 1}_node_{target_node}').index
            weight = tensor[source_node, target_node, layer].item()
            edges.append((source, target))
            weights.append(weight)

# Add edges and weights to the graph
g.add_edges(edges)
g.es['weight'] = weights

# Perform the Leiden algorithm with RBConfigurationVertexPartition, considering edge weights
partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=g.es['weight'])



# Assign colors to communities
palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
g.vs['color'] = [palette[community] for community in partition.membership]

# Define the layout manually (left to right in layers)
layout = []
for layer in range(n_layers):
    for node in range(d_sae):
        layout.append((layer, -node))  # Position each layer in a column

# Plot the graph using matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ig.plot(
    g,
    target=ax,
    layout=layout,
    vertex_size=20,
    vertex_color=g.vs['color'],
    vertex_label=g.vs['name'],
    bbox=(600, 400),
    margin=20
)
plt.savefig('graph_plot.png', dpi=300)
plt.show()

print("Graph plot saved as 'graph_plot.png'")
# %%
