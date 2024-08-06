# Collect explanations for nodes and their most similar upstream neighbors
# for manual inspection of reasonable connections
#
# - Choose X (e.g., 50) random features from any layers (not the first one)
# - For all similarity measures, find the most similar upstream neighbors
# - using some clustering algorithm (e.g. k-means)
# - Create a multi-digraph from these features
# - Fetch all respective explanations from Neuronpedia
# - Create a markdown file which clearly lists features and their inputs
#   together with the explanations


# %%
# Imports
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data, get_filename
from explanation_helpers import find_high_similarity_cluster, add_features_to_graph, connect_features_in_graph, add_explanations


# %%
# Create list of random downstream features
n_layers = 12
d_sae = 24576
number_of_downstream_features = 5

layers = np.random.randint(1, n_layers, size=number_of_downstream_features)
features = np.random.randint(0, d_sae, size=number_of_downstream_features)

downstream_features = list(zip(layers, features))
print(f'Downstream features ({number_of_downstream_features}): {[f"{layer}_{feature}" for layer, feature in downstream_features]}')


# %%
# Loop over all similarity measures and build graph
artefacts_folder = f'../../artefacts/similarity_measures'
measures = ['pearson_correlation', 'jaccard_similarity', 'mutual_information'] # , 'sufficiency', 'necessity']
clamping_thresholds = [0.1, 0.1, 0.1, 0.2, 0.2]
filenames = [f'res_jb_sae_feature_similarity_{measure}_1M_0.0_{clamping_threshold}' for measure, clamping_threshold in zip(measures, clamping_thresholds)]

graph = nx.MultiDiGraph()
for measure, filename in tqdm(zip(measures, filenames), total=len(measures)):
    
    similarities = np.load(f'{artefacts_folder}/{measure}/{filename}.npz')['arr_0']

    for layer, feature in downstream_features:
        graph.add_node(f'{layer}_{feature}', layer=layer, feature=feature, is_downstream=True)

        cluster_values, cluster_indices = find_high_similarity_cluster(similarities[layer - 1, :, feature])

        if len(cluster_values) <= 10:
            add_features_to_graph(graph, layer - 1, cluster_indices)
            connect_features_in_graph(graph, layer, cluster_indices, feature, cluster_values, measure)

    del similarities

# Draw graph for information
nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key='layer'), node_size=10)

# Add explanations to all graph nodes
add_explanations(graph)


# Create markdown file with structured explanations
def format_neuronpedia_link(text, layer, feature):
    return f'[{text}](https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feature})'


output = '# Upstream explanations\n'
for node, attr in graph.nodes(data=True):
    if attr.get('is_downstream', False):
        preds = list(graph.predecessors(node))
        
        layer, feature = graph.nodes[node]['layer'], graph.nodes[node]['feature']
        output += f'## Feature {node} (' + format_neuronpedia_link(graph.nodes[node]["explanation"], layer, feature) + ')\n'

        in_edges = graph.in_edges(node, data=True)
        predecessors_per_measure = [[(from_, attr) for from_, to_, attr in in_edges if attr['measure'] == measure] for measure in measures]

        for measure, predecessors in zip(measures, predecessors_per_measure):
            if not predecessors:
                output += f'### {measure} (No predecessors):\n'
            else:
                output += f'### {measure} ({len(predecessors)} predecessors):\n'
                for predecessor in predecessors:
                    pred_node, pred_attr = predecessor
                    layer, feature = graph.nodes[pred_node]['layer'], graph.nodes[pred_node]['feature']

                    output += f'- {pred_node} (' + format_neuronpedia_link(graph.nodes[pred_node]["explanation"], layer, feature) + f'): {pred_attr["similarity"]:.3f}\n'

# Save file with current date/time since there are no other identifiers
with open(f'../../artefacts/upstream_explanations/{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}_upstream_explanations_{number_of_downstream_features}.md', 'w') as f:
    f.write(output)


# %%
# Save graph to file
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        # Add more conditions here for other numpy types, if needed
        return super(NumpyEncoder, self).default(obj)

with open('../../artefacts/upstream_explanations/sample_graph.json', 'w') as f:
    json.dump(nx.node_link_data(graph), f, cls=NumpyEncoder)
