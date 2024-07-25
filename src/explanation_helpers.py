# %%
# Imports
from typing import List, Optional
from jaxtyping import Int, Float
from collections.abc import Iterable

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import requests
import concurrent.futures
from tqdm import tqdm


# %%
def find_high_similarity_cluster(similarity_values: Float[np.ndarray, "n_features "], method: str = 'kmeans') -> tuple[Float[list, 'n_similar_features'], Int[list, 'n_similar_features']]:
    assert method in ['kmeans']
    
    if method == 'kmeans':
        similarity_values = similarity_values.reshape(-1, 1)
        
        clustering = KMeans(n_clusters=2, random_state=0).fit(similarity_values)
        
        labels = clustering.labels_
        cluster_indices = np.where(labels == 1)[0]
        cluster_values = similarity_values[cluster_indices].flatten()
        
        return list(cluster_values), list(cluster_indices)
    else:
        # TODO: Implement alternative methods
        pass


def add_features_to_graph(graph: nx.MultiDiGraph, layers: int | List[int], features: List[int]) -> None:
    if not isinstance(layers, Iterable):
        layers = [layers] * len(features)

    for layer, feature in zip(layers, features):
        graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature)


def connect_features_in_graph(graph: nx.MultiDiGraph, layer: int, out_features: List[int], in_feature: int, similarities: List[float], measure: str) -> None:
    for out_feature, similarity in zip(out_features, similarities):
        graph.add_edge(f'{layer-1}_{out_feature}', f'{layer}_{in_feature}', similarity=similarity, measure=measure)


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