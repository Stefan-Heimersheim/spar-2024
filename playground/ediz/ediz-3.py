import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
import random
import gc

def load_graph_from_npz(npz_file: str, threshold: float = 0.0) -> nx.DiGraph:
    """
    Loads a graph from a .npz file containing a 3D numpy array.
    The array represents edge weights between nodes across layers.

    Args:
        npz_file: Path to the .npz file.
        threshold: Minimum edge weight to include an edge in the graph.

    Returns:
        A directed graph (DiGraph) with nodes and edges constructed from the array.
    """
    data = np.load(npz_file)
    array = data['arr_0']  # Assuming the array is stored under 'arr_0'
    num_layer_pairs, d_sae, _ = array.shape

    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(num_layer_pairs + 1):
        for j in range(d_sae):
            G.add_node((i, j), layer=i, feature=j)

    # Add edges to the graph based on the array values and threshold
    for i in range(num_layer_pairs):
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


def draw_random_partition(graph: nx.Graph, partition: list[set], num_layers: int, 
                          num_features: int, output_file: str = 'random_partition_graph.png') -> None:
    """
    Draws and saves a plot of a random partition subgraph.

    Args:
        graph: The original graph from which the partition was derived.
        partition: The partitioning of the graph.
        num_layers: The number of layers in the graph.
        num_features: The number of features in each layer.
        output_file: The filename where the plot image will be saved.
    """
    # random_partition = random.choice(partition)
    random_partition = partition[-1] # Let's be deterministic for now!
    subgraph = graph.subgraph(random_partition)
    
    # Use the custom plotting function to visualize the subgraph
    plot_nx_graph_with_mpl(subgraph, num_layers=num_layers, num_features=num_features, filename=output_file)

def main(npz_file: str, threshold: float = 0.0) -> None:
    """
    Main function to load the graph, perform Louvain partitioning, and visualize the results.

    Args:
        npz_file: Path to the .npz file containing the graph data.
        threshold: Minimum edge weight to include an edge in the graph.
    """
    graph = load_graph_from_npz(npz_file, threshold)
    partition = perform_louvain_partition(graph)

    plot_partition_histogram(partition)
    
    num_layers = max(nx.get_node_attributes(graph, 'layer').values()) + 1
    num_features = len(set(attr['feature'] for _, attr in graph.nodes(data=True)))
    
    draw_random_partition(graph, partition, num_layers, num_features)

# Example usage
if __name__ == "__main__":
    npz_file = 'test_graph.npz'  # Replace with your file path
    threshold = 0.1  # Replace with your desired threshold
    main(npz_file, threshold)
