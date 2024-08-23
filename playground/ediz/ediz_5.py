"""
This script performs the following operations on a graph represented by a 3D numpy array loaded from a .npz file:

1. **Loading the Graph**:
   - The script first loads a graph from a `.npz` file using the `load_graph_from_npz` function. 
   - The file is expected to contain a 3D numpy array, where each entry represents an edge weight between nodes in different layers of the graph.
   - Nodes are added to the graph with `layer` and `feature` attributes, and edges are added based on the array values, filtered by a specified weight threshold.

2. **Community Detection**:
   - The `perform_louvain_partition` function applies the Louvain algorithm to detect communities within the graph. 
   - It returns a list of sets, where each set contains the nodes belonging to a particular community.

3. **Visualization**:
   - A histogram of the community sizes is generated using the `plot_partition_histogram` function. 
   - The histogram displays the distribution of node counts across the detected partitions, and the total number of partitions is included in the title.
   - The `draw_random_partition` function selects a specific partition (currently the last one for deterministic behavior) and visualizes it as a subgraph. 
   - The subgraph is saved as a PNG file with nodes arranged by layer and feature, and the layers are labeled at the top.

4. **Main Function**:
   - The `main` function orchestrates the process by loading the graph, performing community detection, generating the histogram, and drawing the partition subgraph.
   - The script allows users to specify the path to the `.npz` file and the edge weight threshold as inputs.

5. **Example Usage**:
   - The script can be executed directly, where it will load a graph from `test_graph.npz`, apply a threshold of 0.1, and proceed with community detection and visualization.

### Functions:
- `load_graph_from_npz(npz_file: str, threshold: float) -> nx.DiGraph`:
  Loads a graph from a `.npz` file, with edges filtered by the specified threshold.

- `perform_louvain_partition(graph: nx.Graph) -> list[set]`:
  Applies the Louvain community detection algorithm to the graph and returns the detected communities.

- `plot_partition_histogram(partition: list[set], output_file: str)`:
  Plots and saves a histogram of the sizes of the detected communities.

- `plot_nx_graph_with_mpl(G: nx.Graph, num_layers: int, num_features: int, filename: str, threshold: float | None)`:
  Visualizes the graph as a PNG file, arranging nodes by layer and feature, and labels the layers at the top.

- `draw_random_partition(graph: nx.Graph, partition: list[set], num_layers: int, num_features: int, output_file: str)`:
  Draws and saves a subgraph for a selected partition.

- `main(npz_file: str, threshold: float)`:
  The entry point of the script, managing the overall process of loading, partitioning, and visualization.

### Example Usage:
```bash
python script_name.py
"""

import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
import random
import gc
import os
import pickle
import leidenalg as la
import igraph as ig

def save_graph(graph: nx.Graph, graph_file: str) -> None:
    """
    Saves the graph to a file.

    Args:
        graph: The graph to save.
        graph_file: Path to the file where the graph will be saved.
    """
    with open(graph_file, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(graph_file: str) -> nx.Graph:
    """
    Loads the graph from a file.

    Args:
        graph_file: Path to the file where the graph is saved.

    Returns:
        The loaded graph.
    """
    with open(graph_file, 'rb') as f:
        return pickle.load(f)

def load_graph_from_npz(npz_file: str, threshold: float = 0.0) -> nx.DiGraph:
    """
    Loads a graph from a .npz file containing a 3D numpy array.
    The array represents edge weights between nodes across layers.

    Args:
        npz_file: Path to the .npz file.
        threshold: Minimum edge weight to include an edge in the graph.

    Returns:
        A directed graph (DiGraph) with nodes and edges constructed from the array.
        If the file does not exist, returns an empty graph.
    """
    if not os.path.exists(npz_file):
        print(f"Error: The file '{npz_file}' does not exist.")
        return nx.DiGraph()
    print(f"Loading: {npz_file}")
    data = np.load(npz_file)
    print("Finished!")
    array = data['arr_0']  # Assuming the array is stored under 'arr_0'
    print(1)
    num_layer_pairs, d_sae, _ = array.shape

    print(f"Found np array of shape: {array.shape}")

    G = nx.DiGraph()

    print("Adding nodes to graph")

    # Add nodes to the graph
    for i in range(num_layer_pairs + 1):
        for j in range(d_sae):
            G.add_node((i, j), layer=i, feature=j)

    print("Adding edges to graph")

    # Add edges to the graph based on the array values and threshold
    for i in range(num_layer_pairs):
        print(f"Making layer:{i}")
        for j in range(d_sae):
            for k in range(d_sae):
                weight = array[i, j, k]
                if weight > threshold:
                    G.add_edge((i, j), (i + 1, k), weight=weight)

    # Free memory by deleting the array and triggering garbage collection
    del array
    gc.collect()

    return G

def perform_louvain_partition(graph: nx.Graph) -> list[set]:
    """
    Performs Louvain community detection on the graph.

    Args:
        graph: The graph on which to perform community detection.

    Returns:
        A list of sets, where each set contains nodes belonging to the same partition.
    """
    return louvain_communities(graph, seed=42)

def perform_leiden_partition(graph: nx.Graph, quality_function=la.CPMVertexPartition) -> list[set]:
    """
    Performs Leiden community detection on the graph using a specified quality function.
    If the graph has edge weights, they will be used in the partitioning process.

    Args:
        graph: The graph on which to perform community detection.
        quality_function: The quality function used for the Leiden algorithm. 
                          Defaults to la.CPMVertexPartition.

    Returns:
        A list of sets, where each set contains nodes belonging to the same partition.
    """
    # Convert the NetworkX graph to an iGraph graph
    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(list(graph.nodes()))
    ig_graph.add_edges(list(graph.edges()))

    # Extract weights if they exist
    if nx.get_edge_attributes(graph, 'weight'):
        print("Using weights for leiden")
        weights = list(nx.get_edge_attributes(graph, 'weight').values())
    else:
        print("Not using weights for leiden")
        weights = None  # No weights, will proceed without them

    # Perform Leiden community detection using the specified quality function and weights
    partition = la.find_partition(ig_graph, quality_function, weights=weights, seed=42)

    # Convert the result back to a list of sets of nodes
    communities = [set(ig_graph.vs[community].indices) for community in partition]

    return communities

def plot_partition_histogram(partition: list[set], output_file: str = 'partition_histogram.png') -> None:
    """
    Plots a histogram of the partition sizes and saves it to a file.
    The total number of partitions is included in the title.

    Args:
        partition: A list of sets representing the partitioned communities.
        output_file: The filename where the histogram image will be saved.
    """
    partition_sizes = [len(part) for part in partition]
    total_partitions = len(partition)  # Calculate the total number of partitions

    plt.figure(figsize=(10, 6))
    plt.hist(partition_sizes, bins=range(1, max(partition_sizes) + 2), edgecolor='black')
    plt.title(f'Distribution of Partitions by Node Count (Total Partitions: {total_partitions})')
    plt.xlabel('Number of Nodes in Partition')
    plt.ylabel('Frequency')
    plt.savefig(output_file)
    plt.close()

def plot_nx_graph_with_mpl(G: nx.Graph, num_layers: int = 12, num_features: int = 5, 
                           filename: str = "plot.png", threshold: float | None = None) -> None:
    """
    Saves a NetworkX graph to a PNG file using Matplotlib, with nodes arranged by layer and feature.
    Also, labels the layers at the top of the graph.

    Args:
        G: The graph to plot.
        num_layers: The number of layers in the graph.
        num_features: The number of features in each layer.
        filename: The filename where the plot image will be saved.
        threshold: If provided, only edges with a weight above this threshold will be drawn.
    """
    pos = {}
    layer_height = 1
    layer_spacing = 10

    # Identify actual layers and features present in the graph
    layers_in_graph = sorted(set(nx.get_node_attributes(G, 'layer').values()))
    features_in_graph = set(nx.get_node_attributes(G, 'feature').values())

    # Generate positions for only the nodes that exist based on their layer and feature
    for i, layer in enumerate(layers_in_graph):
        for feature in range(num_features):
            if feature in features_in_graph:
                pos[(layer, feature)] = (i * layer_spacing, feature * layer_height)

    # Set up the figure size for plotting
    plt.figure(figsize=(int(num_layers * 2.5), int(num_features * .25) + 1))

    # Plot the graph with or without filtering edges by the threshold
    if threshold is not None:
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='blue')
        nx.draw_networkx_edges(G, pos, edgelist=filtered_edges, edge_color='black', arrows=False)
    else:
        nx.draw(G, pos, node_size=100, edge_color='lightgrey', alpha=0.5, node_color="blue", arrows=False)

    # Add labels for the layers that are actually present
    for i, layer in enumerate(layers_in_graph):
        plt.text(i * layer_spacing, (num_features - 1) * layer_height + 1, f"Layer {layer}", 
                 horizontalalignment='center', verticalalignment='bottom', fontsize=12, color='black')

    plt.title("Layered Graph Visualization")
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the figure to a file
    plt.close()  # Close the plot to free up memory

def draw_all_partitions(graph: nx.Graph, partition: list[set], num_layers: int, 
                        num_features: int, output_folder: str = 'partition_graphs') -> None:
    """
    Draws and saves a plot of all partition subgraphs. Each partition is saved as a separate image file
    in a newly created folder.

    Args:
        graph: The original graph from which the partitions were derived.
        partition: The partitioning of the graph.
        num_layers: The number of layers in the graph.
        num_features: The number of features in each layer.
        output_folder: The folder where the partition images will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, part in enumerate(partition):
        subgraph = graph.subgraph(part)
        output_file = os.path.join(output_folder, f'partition_{idx + 1}.png')
        
        # Use the custom plotting function to visualize the subgraph
        plot_nx_graph_with_mpl(subgraph, num_layers=num_layers, num_features=num_features, filename=output_file)

    print(f"All partition graphs have been saved to the folder '{output_folder}'")

def main(npz_file: str, graph_file: str, threshold: float = 0.0) -> None:
    """
    Main function to load the graph, perform Louvain partitioning, and visualize the results.

    Args:
        npz_file: Path to the .npz file containing the graph data.
        graph_file: Path to the file where the graph will be saved/loaded.
        threshold: Minimum edge weight to include an edge in the graph.
    """
    # Check if the graph file exists
    if os.path.exists(graph_file):
        print(f"Loading graph from {graph_file}...")
        graph = load_graph(graph_file)
    else:
        print(f"Graph file not found. Creating graph from {npz_file}...")
        graph = load_graph_from_npz(npz_file, threshold)
        save_graph(graph, graph_file)
        print(f"Graph saved to {graph_file}.")

    print("Performing louvain partition")
    partition = perform_louvain_partition(graph)

    print("Plotting partition histogram")
    plot_partition_histogram(partition)
    
    num_layers = max(nx.get_node_attributes(graph, 'layer').values()) + 1
    num_features = len(set(attr['feature'] for _, attr in graph.nodes(data=True)))
    
    print("Drawing all partitions")
    draw_all_partitions(graph, partition, num_layers, num_features, output_folder="pearson_louvain_0.1")

    print("Done!")

# Example usage
if __name__ == "__main__":
    npz_file = 'np_arrays/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz'  # Replace with your file path
    graph_file = 'graphs/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.pkl'  # Replace with your desired graph file path
    threshold = 0.1  # Replace with your desired threshold
    main(npz_file, graph_file, threshold)
