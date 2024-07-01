# Given a set of correlation measures and correlation matrices for
# each of these measures, do two things:
# 1. Generate a causal graph for each measure, either by taking all
#    connections which are above a given threshold, or by taking a 
#    fixed number of connection between any pair of layers.
# 2. Compute the (pair-wise) overlap between the edge sets of these
#    graphs, i.e., the similarity between the respective measures.

# %%
# Imports
import torch
import networkx as nx
import plotly.express as px
from itertools import combinations


# %%
# Generate dummy connection matrices
n_layers = 5
d_sae = 100

layers = list(range(n_layers))
for layer_1, layer_2 in zip(layers, layers[1:]):
    correlations = torch.rand(d_sae, d_sae)

    correlation_types = ['pearson', 'cosine', 'jaccard', 'mutual_info']
    for type in correlation_types:
        noisy_correlations = correlations + torch.rand(d_sae, d_sae) * 0.005
        with open(f'{type}_{layer_1}_{layer_2}.pt', 'wb') as f:
            torch.save(noisy_correlations, f)


# %%
# Define functions to compute confusion matrices and overlap from graphs

def compute_confusion_matrix(graph1, graph2):
    edges1 = set(frozenset(edge) for edge in graph1.edges())
    edges2 = set(frozenset(edge) for edge in graph2.edges())
    
    yy = len(edges1 & edges2)  # yy = yes/yes = edges is in both graphs
    yn = len(edges1 - edges2)  # yn = yes/no
    ny = len(edges2 - edges1)  # ny = no/yes
    nn = 0  # nn is not typically calculated for sparse graphs due to the large number of possible non-edges
    
    return yy, yn, ny, nn


def create_pairwise_confusion_matrices(graphs):
    for graph in graphs:
        if graph.nodes != graphs[0].nodes:
            raise ValueError('All graphs must have the same node set!')
    
    n = len(graphs)
    confusion_matrices = torch.empty(n, n, 4)

    # Diagonals
    for i in range(n):
        confusion_matrices[i, i] = torch.tensor([graphs[i].number_of_edges(), 0, 0, (graphs[i].number_of_nodes() ** 2) - graphs[i].number_of_edges()])
    
    # Off-diagonals
    for i, j in combinations(range(n), 2):
        yy, yn, ny, nn = compute_confusion_matrix(graphs[i], graphs[j])

        confusion_matrices[i, j] = torch.tensor([yy, yn, ny, nn])
        confusion_matrices[j, i] = torch.tensor([yy, yn, ny, nn])  # Symmetric matrix with flipped yn and ny
    
    return confusion_matrices


def compute_overlap(confusion_matrices, type='absolute'):
    absolute_overlap = confusion_matrices[:, :, 0]

    if type == 'absolute':
        return absolute_overlap
    else:
        relative_overlap = absolute_overlap / confusion_matrices[:, :, 0:3].sum(dim=-1)

        return relative_overlap
    

# %%
def build_causal_graph(n_layers, d_sae, connection_file_naming_fn, threshold=None, k=None):    
    if (threshold is None) == (k is None):
        raise ValueError('Please specify either threshold or k, but not both!')
    
    # Create graph and add nodes for all layers
    graph = nx.Graph()
    graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer}) for layer in range(n_layers) for feature in range(d_sae)])
    
    # For each pair of layers, load connection matrix
    for layer in range(n_layers - 1):
        with open(connection_file_naming_fn(layer), 'rb') as f:
            connections = torch.load(f)

            # Identify edges and add them to graph
            if threshold is not None:
                edge_indices = (connections > threshold).nonzero().tolist()

                graph.add_edges_from([(f'{layer}_{f1}', f'{layer + 1}_{f2}') for f1, f2 in edge_indices])
            else: # k is not None
                _, indices = torch.topk(connections.flatten(), k)
                edge_indices = torch.unravel_index(indices, connections.shape)

                graph.add_edges_from([(f'{layer}_{f1}', f'{layer + 1}_{f2}') for f1, f2 in zip(*edge_indices)])

    return graph


# %%
n_layers = 5
d_sae = 100
correlation_types = ['pearson', 'cosine', 'jaccard', 'mutual_info']
graphs = []

for type in correlation_types:
    graphs.append(build_causal_graph(n_layers, d_sae, lambda layer: f'{type}_{layer}_{layer + 1}.pt', k=10))

confusion_matrices = create_pairwise_confusion_matrices(graphs)
overlap = compute_overlap(confusion_matrices, type='relative')

px.imshow(overlap, zmin=0, zmax=1, x=correlation_types, y=correlation_types)
