import networkx as nx
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import random


def generate_random_multipartite_graph(layer_sizes: list[int]) -> nx.Graph:
    """
    Generates a random multipartite graph using networkx.

    Parameters:
    layer_sizes (list[int]): A list of integers where each integer represents 
                             the number of nodes in each layer.

    Returns:
    nx.Graph: A networkx Graph object representing the generated multipartite graph.
    """
    G = nx.Graph()
    layers = []

    for i, size in enumerate(layer_sizes):
        layer_nodes = [f"L{i}_N{j}" for j in range(size)]
        G.add_nodes_from(layer_nodes, layer=i)
        layers.append(layer_nodes)

    # Add random edges between layers
    for i in range(len(layers) - 1):
        for node1 in layers[i]:
            for node2 in layers[i + 1]:
                if random.random() < 0.5:  # 50% chance of edge creation
                    G.add_edge(node1, node2, weight=random.random())

    return G


def convert_nx_to_igraph(nx_graph: nx.Graph) -> ig.Graph:
    """
    Converts a networkx graph to an igraph graph.

    Parameters:
    nx_graph (nx.Graph): A networkx Graph object.

    Returns:
    ig.Graph: An igraph Graph object with the same structure as the input networkx graph.
    """
    mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in nx_graph.edges()]
    g = ig.Graph(edges=edges, directed=False)
    g.vs['name'] = [reverse_mapping[i] for i in range(len(mapping))]
    g.es['weight'] = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges()]
    g.vs['layer'] = [nx_graph.nodes[node]['layer'] for node in nx_graph.nodes()]
    return g


class PathQualityPartition(la.VertexPartition.MutableVertexPartition):
    def __init__(self, graph: ig.Graph, initial_membership: list[int] = None):
        super().__init__(graph, initial_membership=initial_membership)
        self.weights = graph.es["weight"] if "weight" in graph.es.attributes() else [1] * graph.ecount()
        self._partition = initial_membership if initial_membership else list(range(graph.vcount()))

    def quality(self) -> float:
        total_quality = 0.0
        for community in self.communities:
            community_weight = sum(self.weights[e.index] for v in community for e in self._graph.vs[v].all_edges())
            total_quality += community_weight
        return total_quality

def visualize_partition(nx_graph: nx.Graph, partition_membership: list[int], title: str, file_name: str) -> None:
    """
    Visualizes the graph's partition by coloring nodes according to their partition.

    Parameters:
    nx_graph (nx.Graph): The networkx Graph object.
    partition_membership (list[int]): The partition membership of each node.
    title (str): The title for the plot.
    file_name (str): The name of the file to save the plot as a PNG.
    """
    # Create a color map from the partition membership
    color_map = [partition_membership[i] for i in range(len(nx_graph.nodes()))]

    # Generate the layout for the multipartite graph
    pos = nx.multipartite_layout(nx_graph, subset_key="layer")

    # Draw the graph with node coloring based on partitions
    plt.figure(figsize=(10, 8))
    nx.draw(nx_graph, pos, node_color=color_map, with_labels=True, node_size=500, cmap=plt.cm.viridis)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()


def main() -> None:
    """
    Main function to generate a random multipartite graph, run the Leiden algorithm with both 
    modularity and the custom quality function, and visualize the partitions.
    """
    print("Generating a random multipartite graph...")
    layer_sizes = [10, 10, 10]  # 3 layers with 10 nodes each
    nx_graph = generate_random_multipartite_graph(layer_sizes)
    print("Graph generated.")

    print("Converting networkx graph to igraph format...")
    ig_graph = convert_nx_to_igraph(nx_graph)
    print("Conversion complete.")

    print("Running Leiden algorithm with standard modularity partitioning...")
    partition_modularity = la.find_partition(ig_graph, la.VertexPartition.ModularityVertexPartition)
    print(f"Modularity partitioning complete. Number of partitions: {len(set(partition_modularity.membership))}")
    visualize_partition(nx_graph, partition_modularity.membership, "Modularity Partition", "modularity_partition.png")
    print("Modularity partition visualization saved as 'modularity_partition.png'.")

    print("Running Leiden algorithm with custom path-based quality function...")
    partition_custom = la.find_partition(ig_graph, PathQualityPartition)
    print(f"Custom partitioning complete. Number of partitions: {len(set(partition_custom))}")
    visualize_partition(nx_graph, partition_custom.membership, "Custom Path Quality Partition", "custom_partition.png")
    print("Custom partition visualization saved as 'custom_partition.png'.")

    print("All steps completed successfully.")


if __name__ == "__main__":
    main()
