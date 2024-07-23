# %%
# Imports
from typing import Optional, List

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import requests
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_correlation_data, get_filename


# %%
# Load (clamped) similarity matrix
measure_name = 'pearson_correlation'
folder = f'../../artefacts/similarity_measures/{measure_name}'
filename = get_filename(measure_name, 0.0, 0.1, '1M')
similarities = np.load(f'{folder}/{filename}.npz')['arr_0']


# %%
# Look for dead end features (defined as: The maximum outgoing
# similarity is below a given threshold value) in all layers
threshold = 0.3

def find_indices_below_threshold(arr, threshold):
    result = []
    for i in range(arr.shape[0]):
        slice_max = np.max(arr[i], axis=-1)
        indices_below_threshold = np.where(slice_max <= threshold)[0].tolist()
        result.append(indices_below_threshold)
    
    return result

indices = find_indices_below_threshold(similarities, threshold)

print("Number of dead-end features per layer:")
print([len(ind) for ind in indices])

# TODO: Find dead-end features which are not dead themselves


# %%
# Look for features with a low number of highly similar previous-layer features


# %%
# Playground
layer = 0
feature_1 = 0

sims = similarities[layer, feature_1].reshape(-1, 1)


# %%
# Use k-means with k=2 to identify a cluster of high similarity values
def find_cluster_close_to_one(numbers, eps=0.1, min_samples=2):
    X = numbers.reshape(-1, 1)
    
    clustering = KMeans(n_clusters=2, random_state=0).fit(X)
    
    labels = clustering.labels_
    unique_labels = set(labels)
    best_cluster = -1
    best_mean = -np.inf
    
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        
        cluster_points = X[labels == label]
        cluster_mean = np.mean(cluster_points)
        
        if cluster_mean > best_mean and cluster_mean <= 1:
            best_mean = cluster_mean
            best_cluster = label
    
    if best_cluster == -1:
        return [], []
    
    # Extract indices and values of the best cluster
    cluster_indices = np.where(labels == best_cluster)[0]
    cluster_values = X[cluster_indices].flatten()
    
    return list(cluster_values), list(cluster_indices), best_cluster, unique_labels


# %%
layer = 6
for feature_1 in range(10):
    cluster_values, cluster_indices, best_cluster, unique_labels = find_cluster_close_to_one(similarities[layer, feature_1])

    if len(cluster_values) <= 10:
        print(f'Cluster {best_cluster} has {len(cluster_values)} values: {cluster_values}, {cluster_indices}')
    else:
        print(f'Cluster {best_cluster} has {len(cluster_values)} values.')


# %%
layer = 8
for feature_2 in range(10):
    cluster_values, cluster_indices, best_cluster, unique_labels = find_cluster_close_to_one(similarities[layer, :, feature_2])

    if len(cluster_values) <= 10:
        print(f'Cluster {best_cluster} has {len(cluster_values)} values: {cluster_values}, {cluster_indices}')
    else:
        print(f'Cluster {best_cluster} has {len(cluster_values)} values.')


# %%
# Build a semantic graph:
# - Start at layer X and take the first n features with a low number of high similarities
# - For each of these features, follow the path and find connections which, again, have a low number of high similarities
# - Go until layer Y
# - Build a networkX graph with node labels (layer, feature, explanation) and edge labels (similarity)

layer_from, layer_to = 6, 9
starting_features = [1, 3, 4]

def add_features_to_graph(graph, layer, features):
    for feature in features:
        graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature)

def connect_features_in_graph(graph, layer, out_feature, in_features, similarities):
    for in_feature, similarity in zip(in_features, similarities):
        graph.add_edge(f'{layer}_{out_feature}', f'{layer+1}_{in_feature}', similarity=similarity)

graph = nx.DiGraph()
add_features_to_graph(graph, layer_from, starting_features)

for layer in range(layer_from, layer_to):
    layer_nodes = [(node, attr) for node, attr in graph.nodes(data=True) if attr['layer'] == layer]
    for index, (node, attr) in enumerate(layer_nodes):
        if attr['layer'] == layer:
            print(f'Feature {node} ({index+1}/{len(layer_nodes)} in layer {layer}):')
            cluster_values, cluster_indices, best_cluster, unique_labels = find_cluster_close_to_one(similarities[attr['layer'], attr['feature']])

            if len(cluster_values) <= 10:
                print(f'Cluster {best_cluster} has {len(cluster_values)} values: {cluster_values}, {cluster_indices}')

                add_features_to_graph(graph, layer + 1, cluster_indices)
                connect_features_in_graph(graph, layer, attr['feature'], cluster_indices, cluster_values)


# %%
nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key='layer'), node_size=10)


# %%
# Build a semantic graph from backwards connections:
# - Start at layer X and take the first n features with a low number of high similarities
# - For each of these features, follow the path and find connections which, again, have a low number of high similarities
# - Go until layer Y
# - Build a networkX graph with node labels (layer, feature, explanation) and edge labels (similarity)

layer_from, layer_to = 8, 6
starting_features = list(range(10))

def add_features_to_graph(graph, layer, features):
    for feature in features:
        graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature)

def connect_features_in_graph(graph, layer, in_feature, out_features, similarities):
    for out_feature, similarity in zip(out_features, similarities):
        graph.add_edge(f'{layer-1}_{out_feature}', f'{layer}_{in_feature}', similarity=similarity)

graph = nx.DiGraph()
add_features_to_graph(graph, layer_from, starting_features)

for layer in range(layer_from, layer_to, -1):
    layer_nodes = [(node, attr) for node, attr in graph.nodes(data=True) if attr['layer'] == layer]
    for index, (node, attr) in enumerate(layer_nodes):
        if attr['layer'] == layer:
            print(f'Feature {node} ({index+1}/{len(layer_nodes)} in layer {layer}):')
            cluster_values, cluster_indices, best_cluster, unique_labels = find_cluster_close_to_one(similarities[attr['layer'], :, attr['feature']])

            if len(cluster_values) <= 10:
                print(f'Cluster {best_cluster} has {len(cluster_values)} values: {cluster_values}, {cluster_indices}')

                add_features_to_graph(graph, layer - 1, cluster_indices)
                connect_features_in_graph(graph, layer, attr['feature'], cluster_indices, cluster_values)


# %%
nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key='layer'), node_size=10)


# %%
def get_explanation(layer, feature, model_name='gpt2-small', sae_name='res-jb') -> Optional[str]:
    res = requests.get(f'https://www.neuronpedia.org/api/feature/{model_name}/{layer}-{sae_name}/{feature}').json()
    explanation = res['explanations'][0]['description'] if len(res['explanations']) > 0 else None

    return explanation


def add_explanations(graph: nx.DiGraph) -> None:
    progress_bar = tqdm(graph.nodes(data=True), desc='Adding explanations from Neuronpedia')
    for node, attr in progress_bar:
        layer, feature, explanation = attr['layer'], attr['feature'], attr.get('explanation', None)
        
        # Only request and add explanation if there is none yet
        if explanation is None:
            graph.nodes[node]['explanation'] = get_explanation(layer, feature)    


def print_predecessor_subgraph_explanations(graph: nx.DiGraph, nodes: List[str]) -> None:
    for node in nodes:
        preds = list(graph.predecessors(node))
        
        print(f'Feature {node} ({graph.nodes[node]["explanation"]}) has {len(preds)} predecessors:')

        for pred in preds:
            print(f'- {pred} ({graph.nodes[pred]["explanation"]})')

        print()


# %%
out_features = [node for node in graph.nodes() if len(list(graph.predecessors(node))) > 0]
add_explanations(graph)
print_predecessor_subgraph_explanations(graph, out_features)

