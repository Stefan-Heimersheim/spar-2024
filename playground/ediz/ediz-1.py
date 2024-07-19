# %%
import numpy as np
import torch
import networkx as nx
# %%
d_sae = 50
n_layers = 12
tensor = torch.rand(d_sae, d_sae, n_layers-1)
# %%
from graphing_utils import nx_graph_from_tensor, plot_nx_graph_with_mpl

G = nx_graph_from_tensor(tensor)
plot_nx_graph_with_mpl(G, n_layers, d_sae, threshold=.99)


# %%
import graph_converter as gc

G_ig = gc.networkx_to_igraph(G)

# %%
import graph_algorithms as ga

partition = ga.get_module_partition_leiden(G_ig)
# %%
# Color the networkx graph based on partitions
# Create a color map
color_map = []
colors = ["red", "green", "blue", "yellow", "purple", "orange"]

# Ensure the number of colors matches the number of partitions
if len(colors) < len(partition):
    colors = colors * ((len(partition) // len(colors)) + 1)

for node in G.nodes():
    node_community = partition[G_ig.vs.find(_nx_name=node).index]
    color_map.append(colors[node_community])


# %%
