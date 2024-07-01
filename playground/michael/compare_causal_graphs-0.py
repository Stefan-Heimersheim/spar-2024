# %%
import networkx as nx
import torch
from itertools import combinations


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
graphs = [
    nx.Graph([(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]),
    nx.Graph([(0, 1), (0, 4), (1, 2), (1, 3)]),
    nx.Graph([(0, 2), (0, 4), (1, 4), (2, 4), (3, 4)])
]

confusion_matrices = create_pairwise_confusion_matrices(graphs)
confusion_matrices


# %%
compute_overlap(confusion_matrices, type='relative')