# %%
# Imports
from typing import Optional

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import requests
import concurrent.futures
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_correlation_data, get_filename


# %%
# Load (clamped) similarity matrix
measure_name = 'jaccard_similarity'
folder = f'../../artefacts/similarity_measures/{measure_name}'
filename = get_filename(measure_name, 0.0, 0.1, '1M')
similarities = np.load(f'{folder}/{filename}.npz')['arr_0']


# %%
# First experiment (WIP):
# 
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

# TODO: Identify dead-end features which are not dead themselves


# %%
# Second experiment:
# 
# Look for features with a low number of highly similar previous-layer features,
# starting from a high layer
# - Start at layer_from and take the first n features with a low number of high similarities
# - For each of these features, follow the path and find connections which, again, have a low number of high similarities
# - Go until layer_to
# - Build a networkX graph with node labels (layer, feature, explanation) and edge labels (similarity)

# Use k-means with k=2 to identify a cluster of high similarity values
def find_high_similarity_cluster(values) -> tuple[list[float], list[int]]:
    values = values.reshape(-1, 1)
    
    clustering = KMeans(n_clusters=2, random_state=0).fit(values)
    
    labels = clustering.labels_
    cluster_indices = np.where(labels == 1)[0]
    cluster_values = values[cluster_indices].flatten()
    
    return list(cluster_values), list(cluster_indices)


def add_features_to_graph(graph, layer, features):
    for feature in features:
        graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature)


def connect_features_in_graph(graph, layer, out_features, in_feature, similarities, measure):
    for out_feature, similarity in zip(out_features, similarities):
        graph.add_edge(f'{layer-1}_{out_feature}', f'{layer}_{in_feature}', similarity=similarity, measure=measure)


# %%
layer_from, layer_to = 9, 6

# Start with some features in layer_from
starting_features = []
for feature_2 in range(10):
    cluster_values, cluster_indices = find_high_similarity_cluster(similarities[layer_from, :, feature_2])

    if len(cluster_values) <= 10:
        starting_features.append(feature_2)

graph = nx.DiGraph()
add_features_to_graph(graph, layer_from, starting_features)

print(f'Starting features: {[f"{layer_from}_{feature}" for feature in starting_features]}')

for layer in range(layer_from, layer_to, -1):
    layer_nodes = [(node, attr) for node, attr in graph.nodes(data=True) if attr['layer'] == layer]
    for index, (node, attr) in enumerate(layer_nodes):
        if attr['layer'] == layer:
            print(f'Feature {node} ({index+1}/{len(layer_nodes)} in layer {layer}):')
            cluster_values, cluster_indices = find_high_similarity_cluster(similarities[attr['layer'], :, attr['feature']])

            if len(cluster_values) <= 10:
                print(f'  {len(cluster_values)} highly similar features: {cluster_values}, {cluster_indices}')

                add_features_to_graph(graph, layer - 1, cluster_indices)
                connect_features_in_graph(graph, layer, cluster_indices, attr['feature'], cluster_values)
            else:
                print(f'  (skipped) {len(cluster_values)} highly similar features')


# %%
nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key='layer'), node_size=10)


# %%
# Add Neuronpedia explanations to the graphs and print predecessor subgraphs
# for manual inspection

def get_explanation(layer, feature, model_name='gpt2-small', sae_name='res-jb') -> Optional[str]:
    """Fetches a single explanation for a given layer and feature from Neuronpedia.
    """
    res = requests.get(f'https://www.neuronpedia.org/api/feature/{model_name}/{layer}-{sae_name}/{feature}').json()
    explanation = res['explanations'][0]['description'] if len(res['explanations']) > 0 else None

    return explanation


def add_explanations(graph: nx.DiGraph) -> None:
    """Fetches all Neuronpedia explanations for a given graph of features.
    """
    # progress_bar = tqdm(graph.nodes(data=True), desc='Adding explanations from Neuronpedia')
    # for node, attr in progress_bar:
    #     layer, feature, explanation = attr['layer'], attr['feature'], attr.get('explanation', None)
        
    #     # Only request and add explanation if there is none yet
    #     if explanation is None:
    #         graph.nodes[node]['explanation'] = get_explanation(layer, feature) 
    # 
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_data = {executor.submit(get_explanation, attr['layer'], attr['feature']): node for node, attr in graph.nodes(data=True)}
        for future in tqdm(concurrent.futures.as_completed(future_to_data)):
            node = future_to_data[future]
            if graph.nodes[node].get('explanation') is None:
                try:
                    explanation = future.result()
                    graph.nodes[node]['explanation'] = explanation
                except Exception as e:
                    print(f'Failed for node {node}!')


def print_predecessor_subgraph_explanations(graph: nx.DiGraph, nodes: List[str]) -> None:
    """Gathers predecessor subgraphs for all given nodes and prints the respective
    explanations for manual inspection.
    """
    for node in nodes:
        preds = list(graph.predecessors(node))
        
        print(f'Feature {node} ({graph.nodes[node]["explanation"]}) has {len(preds)} predecessors:')

        for pred in preds:
            print(f'- {pred} ({graph.nodes[pred]["explanation"]})')

        print()

out_features = [node for node in graph.nodes() if len(list(graph.predecessors(node))) > 0]
add_explanations(graph)
print_predecessor_subgraph_explanations(graph, out_features)


# %%
layers = np.random.randint(1, 12, size=50)
features = np.random.randint(0, 24576, size=50)

downstream_features = list(zip(layers, features))
print(f'Downstream features: {[f"{layer}_{feature}" for layer, feature in downstream_features]}')


# %%
graph = nx.MultiDiGraph()
for layer, feature in downstream_features:
    print(f'Feature {layer}_{feature}:')

    graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature)

    cluster_values, cluster_indices = find_high_similarity_cluster(similarities[attr['layer'], :, attr['feature']])

    if len(cluster_values) <= 10:
        print(f'  {len(cluster_values)} highly similar features: {cluster_values}, {cluster_indices}')

        add_features_to_graph(graph, layer - 1, cluster_indices)
        connect_features_in_graph(graph, layer, cluster_indices, feature, cluster_values)
    else:
        print(f'  (skipped) {len(cluster_values)} highly similar features')


# %%
# Loop over all similarity measures
measures = ['pearson_correlation', 'jaccard_similarity', 'mutual_information'] #, 'forward_implication', 'backward_implication']
clamping_thresholds = [0.1, 0.1, 0.3]
filenames = [f'res_jb_sae_feature_similarity_{measure}_1M_0.0_{clamping_threshold}' for measure, clamping_threshold in zip(measures, clamping_thresholds)]

# %%
graph = nx.MultiDiGraph()
for measure, filename in tqdm(zip(measures, filenames), total=len(measures)):
    folder = f'../../artefacts/similarity_measures/{measure}'
    similarities = np.load(f'{folder}/{filename}.npz')['arr_0']

    for layer, feature in downstream_features:
        graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature, is_downstream=True)

        cluster_values, cluster_indices = find_high_similarity_cluster(similarities[layer - 1, :, feature])

        if len(cluster_values) <= 10:
            add_features_to_graph(graph, layer - 1, cluster_indices)
            connect_features_in_graph(graph, layer, cluster_indices, feature, cluster_values, measure)

    del similarities

add_explanations(graph)


# %%
def format_link(text, layer, feature):
    return f'[{text}](https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feature})'


# %%
output = ''
for node, attr in graph.nodes(data=True):
    if attr.get('is_downstream', False):
        preds = list(graph.predecessors(node))
        
        layer, feature = graph.nodes[node]['layer'], graph.nodes[node]['feature']
        output += format_link(f'Feature {node} ({graph.nodes[node]["explanation"]})', layer, feature) + f' has {len(preds)} predecessors:\n\n'

        for measure in measures:
            output += measure + '\n\n'
            for pred in preds:
                layer, feature = graph.nodes[pred]['layer'], graph.nodes[pred]['feature']
                if measure in [attr['measure'] for attr in graph.get_edge_data(pred, node).values()]:
                    output += format_link(f'- {pred} ({graph.nodes[pred]["explanation"]})', layer, feature) + '\n\n'

            output += '\n\n'

        output += '\n-----------------------------------\n'

# %%
console = Console()
console.print(Markdown(output))


# %%
print('\n'.join(str(edge) for edge in graph.in_edges('3_3463', data=True)))


# %%
with open('../../artefacts/upstream_explanations.md', 'w') as f:
    f.write(output)