# %%
# Imports
import plotly.graph_objects as go
import networkx as nx
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from explanation_helpers import find_high_similarity_cluster, add_features_to_graph, connect_features_in_graph, add_explanations
from visualization import show_explanation_graph


# %%
# Create list of random downstream features
n_layers = 12
d_sae = 24576
number_of_downstream_features = 10

layers = np.random.randint(1, n_layers, size=number_of_downstream_features)
features = np.random.randint(0, d_sae, size=number_of_downstream_features)

downstream_features = list(zip(layers, features))


# %%
# Build graph from similarity measure
artefacts_folder = f'../../artefacts/similarity_measures'
measure = 'pearson_correlation'
clamping_threshold = 0.1
filename = f'res_jb_sae_feature_similarity_{measure}_1M_0.0_{clamping_threshold}'

graph = nx.MultiDiGraph()
similarities = np.load(f'{artefacts_folder}/{measure}/{filename}.npz')['arr_0']

for layer, feature in downstream_features:
    graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature, is_downstream=True)

    cluster_values, cluster_indices = find_high_similarity_cluster(similarities[layer - 1, :, feature])

    if len(cluster_values) <= 10:
        add_features_to_graph(graph, layer - 1, cluster_indices)
        connect_features_in_graph(graph, layer, cluster_indices, feature, cluster_values, measure)

# Add explanations to all graph nodes
add_explanations(graph)

# Show graph
show_explanation_graph(graph)
