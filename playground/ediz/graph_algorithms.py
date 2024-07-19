import networkx as nx
import igraph as ig

import leidenalg

def get_module_partition_leiden(G_ig):
    # takes igraph, returns partition
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, weights=G_ig.es['weight'])
    return partition