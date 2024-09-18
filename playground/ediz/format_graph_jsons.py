# %%
import networkx as nx

import numpy as np

import json
import os

from partition_utils import load_graph, read_partition_from_file

def mask_graph(graph: nx.DiGraph, mask_file:str, mask_index:int=0) -> nx.DiGraph:
    mask = np.load(mask_file)['arr_0'][mask_index]
    graph_copy = graph.copy()
    nodes_to_remove = []
    for nodeid,node_data in graph_copy.nodes(data=True):
        if mask[node_data.get('layer')][node_data.get('feature')] == False:
            nodes_to_remove.append(nodeid)
    graph_copy.remove_nodes_from(nodes_to_remove)
    return graph_copy


def convert_weight_to_similarity(G):
    """
    Converts the 'weight' attribute of each edge in the graph to 'similarity'.
    
    Parameters:
    G (networkx.Graph): A NetworkX graph with 'weight' attributes on edges.
    
    Returns:
    networkx.Graph: The graph with 'weight' attributes renamed to 'similarity'.
    """
    # Iterate over all edges in the graph
    for u, v, data in G.edges(data=True):
        # Check if 'weight' is in the edge data
        if 'weight' in data:
            # Copy the weight to similarity
            data['similarity'] = data['weight']
            # Remove the weight attribute
            del data['weight']
    
    return G


def reformat_graph_ids_nx(G):
    """
    Reformats a NetworkX graph's node IDs and edge IDs from lists to strings in the form 'layer_feature'.
    
    Parameters:
    G (networkx.Graph): A NetworkX graph with node IDs as lists like [layer, feature].

    Returns:
    networkx.Graph: A new NetworkX graph with reformatted node IDs and edges.
    """
    # Create a new graph to store the reformatted data
    new_G = nx.DiGraph() if G.is_directed() else nx.Graph()
    
    id_mapping = {}

    # Reformat nodes
    for old_id, data in G.nodes(data=True):
        # Assume old_id is a list in the form [layer, feature]
        layer, feature = old_id
        new_id = f"{layer}_{feature}"
        
        # Map old_id to new_id
        id_mapping[tuple(old_id)] = new_id
        
        # Add node with new_id to the new graph, preserving the data
        new_G.add_node(new_id, **data)
    
    # Reformat edges
    for u, v, data in G.edges(data=True):
        # Map old source and target ids to the new ids
        new_u = id_mapping[tuple(u)]
        new_v = id_mapping[tuple(v)]
        
        # Add edge with new node ids to the new graph, preserving the data
        new_G.add_edge(new_u, new_v, **data)
    
    return new_G

def save_graph_to_json(graph, filename):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.float32):
                return float(obj)
            # Add more conditions here for other numpy types, if needed
            return super(NumpyEncoder, self).default(obj)

    with open(filename, 'w') as f:
        json.dump(nx.node_link_data(graph), f, cls=NumpyEncoder)


def get_subgraph_from_partition(G, partition, index):
    """
    Returns the subgraph corresponding to a specific partition.

    Parameters:
    G (networkx.Graph): The original NetworkX graph.
    partition (list[set]): A list of sets, where each set contains nodes in a partition.
    index (int): The index of the partition to select.

    Returns:
    networkx.Graph: The subgraph containing only the nodes in the specified partition.
    """
    # Select the partition by index
    nodes_in_partition = partition[index]
    
    # Generate the subgraph containing only the nodes in the selected partition
    subgraph = G.subgraph(nodes_in_partition).copy()
    
    return subgraph

def assign_community_labels(graph: nx.Graph, partition: list[set]) -> nx.Graph:
    """
    Assigns a 'community' attribute to each node in the graph, based on the given partition.

    Parameters:
    graph (nx.Graph): The input graph.
    partition (list[set]): A list of sets, where each set contains nodes that belong to the same community.

    Returns:
    nx.Graph: The graph with the 'community' attribute assigned to each node.
    """
    for community_index, community in enumerate(partition):
        for node in community:
            if node in graph:
                graph.nodes[node]['community'] = community_index
            else:
                raise ValueError(f"Node {node} in partition is not present in the graph.")
    
    return graph


def save_not_masked():
    graph_to_save = "graphs/sufficiency/res_jb_sae_feature_similarity_sufficiency_10M_relative_activation_0.2_threshold_0.1.pkl"
    partition_path = "partitions/sufficiency_louvain_threshold_0.1_plotly.pkl"
    partition = read_partition_from_file(partition_path)
    graph = load_graph(graph_to_save)

    partition_index = [i for i,part in enumerate(partition) if (len(part) < 200 and len(part) > 4)]
    print("partition index;", len(partition_index))
    subgraphs = [get_subgraph_from_partition(graph,partition,i) for i in partition_index]
    lengths = [sg.number_of_nodes() for sg in subgraphs]
    print(lengths)
    print(len(partition))

    root_dir = "graph_jsons"
    graphs_dir = "sufficiency_louvain_threshold_0.1"
    output_dir = f"{root_dir}/{graphs_dir}"
    os.makedirs(output_dir, exist_ok=True)
    for i,sg in enumerate(subgraphs):
        print("saved", i)
        graph_size = len(list(sg.nodes()))
        save_graph_to_json(convert_weight_to_similarity(reformat_graph_ids_nx(sg)), f"{output_dir}/{graphs_dir}_size_{graph_size}_{i}.json")




def save_masked():
    graph_to_save = "graphs/necessity/res_jb_sae_feature_similarity_necessity_10M_relative_activation_0.2_threshold_0.1.pkl"
    graph = load_graph(graph_to_save)
    partition_path = "partitions/necessity_leiden_modularity_threshold_0.1_plotly.pkl"
    partition = read_partition_from_file(partition_path)
    mask_file = "../../artefacts/active_features/res_jb_sae_active_features_rel_0.1_100_last.npz"
    mask_index = 0
    graphs = [mask_graph(assign_community_labels(graph,partition),mask_file,i) for i in range(100)]

    root_dir = "graph_jsons"
    graphs_dir = "necessity_leiden_modularity_threshold_0.1_masked_single"
    output_dir = f"{root_dir}/{graphs_dir}"
    os.makedirs(output_dir, exist_ok=True)
    print("graph number of nodes", graph.number_of_nodes())
    print("partition number of nodes", sum([len(s) for s in partition]))

    graphs = [convert_weight_to_similarity(reformat_graph_ids_nx(graph)) for graph in graphs]

    for i,graph in enumerate(graphs):
        save_graph_to_json(graph, f"{output_dir}/{graphs_dir}_{i}.json")

if __name__ == "__main__":
    #save_masked()
    save_not_masked()
