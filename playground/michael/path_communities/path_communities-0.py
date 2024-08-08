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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data
from similarity_helpers import clamp_low_values, save_compressed, load_similarity_data
from explanation_helpers import add_explanations
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
similarities = np.load('../../../artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz')['arr_0']


# %%
n_layers = 12
d_sae = 24576

paths = np.empty((d_sae, n_layers), dtype=int)
paths[:, 0] = np.arange(d_sae)

max_similarity_successors = similarities.argmax(axis=-1)

for layer in range(n_layers - 1):
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
linkage_matrix = linkage(squareform(distances), method='average')



# %%
# See how the number of clusters changes with the distance parameter
plt.plot([fcluster(linkage_matrix, t=t, criterion='distance').max() for t in range(15)])
plt.xlabel('Maximum cluster distance')
plt.ylabel('Number of clusters')
plt.show()


# %%
# Set a distance threshold or number of clusters
t = 11  # This could be a distance threshold or number of clusters, depending on the criterion
criterion = 'distance'  # or 'maxclust' if t represents the number of clusters

clusters = fcluster(linkage_matrix, t=t, criterion=criterion)


# %%
lower_bound, upper_bound = 5, 10
reasonable_size_clusters = [cluster for cluster, size in Counter(clusters).items() if lower_bound <= size <= upper_bound]
print(f'{len(reasonable_size_clusters)=}')


# %%
def show_cluster(paths, cluster_path_indices, show):
    cluster_paths = paths[cluster_path_indices]

    graph = nx.DiGraph()
    for path in cluster_paths:
        graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer, 'feature': feature}) for layer, feature in enumerate(path)])
        graph.add_edges_from([(f'{layer}_{out_feature}', f'{layer+1}_{in_feature}', {'similarity': similarities[layer, out_feature, in_feature]}) for layer, (out_feature, in_feature) in enumerate(zip(path, path[1:]))])

    add_explanations(graph)
    return show_explanation_graph(graph, show=show)


# %%
for i in range(10):
    cluster_path_indices = np.argwhere(clusters == reasonable_size_clusters[i]).flatten()
    fig = show_cluster(paths, cluster_path_indices, show=False)
    fig.update_layout(title=f'Cluster {i} ({len(cluster_path_indices)} paths)')


# %%
fig.write_html('plot.html')