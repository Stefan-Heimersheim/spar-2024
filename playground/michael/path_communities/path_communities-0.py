# %%
# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import sys
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from numba import jit
from collections import Counter
import networkx as nx
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from similarity_helpers import load_similarity_data
from visualization import show_explanation_graph

# %%
# OPTIONAL: Check if the correct GPU is visible
print(torch.cuda.device_count())  # Should print 1
print(torch.cuda.current_device())  # Should print 0 since it's the first visible device
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
measure_name = 'cosine_similarity'
sae_name = 'res_jb_sae'

# Load similarities from unclamped files to avoid clamping errors
similarities = load_similarity_data([f'../../../artefacts/similarity_measures/{measure_name}/.unclamped/{sae_name}_feature_similarity_{measure_name}_{layer}.npz' for layer in range(11)])

# Load explanations
with open(f'../../../artefacts/explanations/{sae_name}_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)


# %%
n_layers = 12
d_sae = 24576

paths = np.empty((d_sae, n_layers), dtype=int)
paths[:, 0] = 0
paths[:, 1] = np.arange(d_sae)

max_similarity_successors = similarities.argmax(axis=-1)

for layer in range(1, n_layers - 1):
    paths[:, layer + 1] = max_similarity_successors[layer, paths[:, layer]]


# %%
@jit(nopython=True)
def fast_equality_distance(seq1, seq2):
    return (seq1 != seq2).sum()

@jit(nopython=True)
def compute_distance_row(i, paths):
    n_paths = paths.shape[0]
    row = np.zeros(n_paths, dtype=np.float32)
    for j in range(i+1, n_paths):
        dist = fast_equality_distance(paths[i], paths[j])
        row[j] = dist
    return row

def fast_pairwise_distances_with_progress(paths):
    n_paths = paths.shape[0]
    distances = np.zeros((n_paths, n_paths), dtype=np.float32)
    
    for i in tqdm(range(n_paths), desc="Computing pairwise distances"):
        row = compute_distance_row(i, paths)
        distances[i, i+1:] = row[i+1:]
        distances[i+1:, i] = row[i+1:]
    
    return distances

# Compute pairwise distances
distances = fast_pairwise_distances_with_progress(paths)

# Perform hierarchical clustering
print('Computing clustering hierarchy...', end='')
linkage_matrix = linkage(squareform(distances), method='average')
print('done.')

# %%
# Show how the number of clusters changes with the distance parameter
plt.plot([fcluster(linkage_matrix, t=t, criterion='distance').max() for t in range(15)])
plt.xlabel('Maximum cluster distance')
plt.ylabel('Number of clusters')
plt.show()


# %%
# Set a distance threshold or number of clusters
t = 10  # This could be a distance threshold or number of clusters, depending on the criterion
criterion = 'distance'  # or 'maxclust' if t represents the number of clusters

clusters = fcluster(linkage_matrix, t=t, criterion=criterion)


# %%
lower_bound, upper_bound = 5, 10
reasonable_size_clusters = [cluster for cluster, size in Counter(clusters).items() if lower_bound <= size <= upper_bound]
print(f'{len(reasonable_size_clusters)=}')


# %%
def add_explanations(graph):
    for node, attr in graph.nodes(data=True):
        graph.nodes[node]['explanation'] = explanations[attr['layer']][attr['feature']]

def show_cluster(paths, cluster_path_indices, show):
    cluster_paths = paths[cluster_path_indices]

    graph = nx.DiGraph()
    for path in cluster_paths:
        graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer, 'feature': feature}) for layer, feature in enumerate(path)])
        graph.add_edges_from([(f'{layer}_{out_feature}', f'{layer+1}_{in_feature}', {'similarity': similarities[layer, out_feature, in_feature]}) for layer, (out_feature, in_feature) in enumerate(zip(path, path[1:]))])

    add_explanations(graph)
    return show_explanation_graph(graph, show=show)


# %%
number_of_cluster_plots = 3

folder = '../../../artefacts/path_clusters'
Path(folder).mkdir(parents=True, exist_ok=True)

for i in range(number_of_cluster_plots):
    cluster_path_indices = np.argwhere(clusters == reasonable_size_clusters[i]).flatten()
    fig = show_cluster(paths, cluster_path_indices, show=False)
    fig.update_layout(title=f'[{t=}, {lower_bound=}, {upper_bound=}] Cluster {i} ({len(cluster_path_indices)} paths)')
    fig.show()

    fig.write_html(f'{folder}/{sae_name}_{measure_name}_path_cluster_{t}_{lower_bound}_{upper_bound}_{i}.html')

# %%
similarities[1, :10, :].max(axis=-1)


# %%
# Inspect cosine similarities
similarities.max()