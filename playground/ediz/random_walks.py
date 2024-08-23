import networkx as nx
import random
from collections import Counter
from multiprocessing import Pool, cpu_count, Value
from ediz_6 import load_graph
import os
from tqdm import tqdm
import pickle


def random_walk(graph: nx.Graph, start_node: int, walk_length: int, weight: str = 'weight') -> list[int]:
    """
    Performs a single random walk starting from the specified start_node.

    Parameters:
    - graph (nx.Graph): The graph on which the random walk is performed.
    - start_node (int): The node from which the random walk starts.
    - walk_length (int): The maximum length of the random walk.
    - weight (str): The edge attribute used as weights for the walk. Defaults to 'weight'.

    Returns:
    - walk (list[int]): A list of nodes representing the path taken during the random walk.
    """
    walk = [start_node]
    current_node = start_node
    
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break  # If there are no neighbors, end the walk
        if weight:
            # Choose the next node based on edge weights
            weights = [graph[current_node][n].get(weight, 1) for n in neighbors]
            next_node = random.choices(neighbors, weights=weights, k=1)[0]
        else:
            # Choose the next node uniformly at random
            next_node = random.choice(neighbors)
        walk.append(next_node)
        current_node = next_node
    
    return walk

def random_walk_worker(args: tuple) -> Counter:
    """
    Worker function to perform a subset of random walks in parallel.

    Parameters:
    - args (tuple): A tuple containing the graph, number of walks, walk length, 
                    weight attribute, start nodes, and valid end nodes.

    Returns:
    - path_counter (Counter): A Counter object that counts the frequency of paths taken.
    """
    graph, num_walks, walk_length, weight, start_nodes, valid_end_nodes = args
    path_counter = Counter()
    
    for _ in range(num_walks):
        for node in start_nodes:
            walk = random_walk(graph, node, walk_length, weight)
            
            # Only consider paths that start in start_layer and end in end_layer
            if walk[-1] in valid_end_nodes:
                for i in range(1, len(walk)):
                    path = tuple(walk[:i+1])
                    path_counter[path] += 1
    
    return path_counter

def perform_random_walks_parallel(graph: nx.Graph, num_walks: int, walk_length: int, 
                                  weight: str = 'weight', start_layer: int = None, 
                                  end_layer: int = None) -> Counter:
    """
    Performs random walks from nodes in the start_layer and collects paths that end in the end_layer.
    This function parallelizes the computation across multiple CPUs.
    """
    # Filter nodes by layer
    start_nodes = [node for node, data in graph.nodes(data=True) if data.get('layer') == start_layer]
    valid_end_nodes = {node for node, data in graph.nodes(data=True) if data.get('layer') == end_layer}
    
    # Determine the number of CPUs to use
    num_cpus = cpu_count()
    
    # Divide the work among the CPUs
    walks_per_cpu = num_walks // num_cpus
    remainder_walks = num_walks % num_cpus
    
    # Prepare arguments for each worker process
    args = []
    for i in range(num_cpus):
        walks_for_this_cpu = walks_per_cpu + (1 if i < remainder_walks else 0)
        args.append((graph, walks_for_this_cpu, walk_length, weight, start_nodes, valid_end_nodes))
    
    # Execute the random walks in parallel
    with Pool(processes=num_cpus) as pool:
        results = pool.map(random_walk_worker, args)
    
    # Combine the results from all workers
    combined_counter = Counter()
    for i, result in enumerate(results):
        print(f"Worker {i+1} completed.")
        combined_counter.update(result)
    
    return combined_counter

def extract_significant_paths(path_counter: Counter, min_freq: int = 2) -> list[set[int]]:
    """
    Extracts significant paths based on frequency.

    Parameters:
    - path_counter (Counter): A Counter object containing the frequency of paths.
    - min_freq (int): The minimum frequency for a path to be considered significant. Defaults to 2.

    Returns:
    - significant_paths (list[set[int]]): A list of sets, where each set represents a significant path.
    """
    significant_paths = [set(path) for path, freq in path_counter.items() if freq >= min_freq]
    
    # Post-process to remove redundant paths
    significant_paths = merge_overlapping_paths(significant_paths)
    
    return significant_paths

def merge_overlapping_paths(paths: list[set[int]]) -> list[set[int]]:
    """
    Merges overlapping paths into larger path structures.

    Parameters:
    - paths (list[set[int]]): A list of sets, where each set represents a path.

    Returns:
    - merged_paths (list[set[int]]): A list of merged path sets, removing redundancy.
    """
    merged_paths = []
    
    while paths:
        current_path = paths.pop()
        merged = False
        
        for i, other_path in enumerate(merged_paths):
            if current_path & other_path:  # Check if there is any intersection
                merged_paths[i] = current_path | other_path  # Merge paths
                merged = True
                break
        
        if not merged:
            merged_paths.append(current_path)
    
    return merged_paths

def find_path_like_structures(graph: nx.Graph, num_walks: int = 100, walk_length: int = 10, 
                              min_freq: int = 2, weight: str = 'weight', start_layer: int = None, 
                              end_layer: int = None) -> list[set[int]]:
    """
    Main function to find path-like structures in the multipartite graph, with parallel processing.

    Parameters:
    - graph (nx.Graph): The graph on which the random walks are performed.
    - num_walks (int): The total number of random walks to be performed. Defaults to 100.
    - walk_length (int): The maximum length of each random walk. Defaults to 10.
    - min_freq (int): The minimum frequency for a path to be considered significant. Defaults to 2.
    - weight (str): The edge attribute used as weights for the walk. Defaults to 'weight'.
    - start_layer (int): The layer from which the random walks should start. Defaults to None.
    - end_layer (int): The layer in which the paths should end. Defaults to None.

    Returns:
    - significant_paths (list[set[int]]): A list of sets, where each set represents a significant path.
    """
    path_counter = perform_random_walks_parallel(graph, num_walks, walk_length, weight, start_layer, end_layer)
    significant_paths = extract_significant_paths(path_counter, min_freq)
    
    return significant_paths

def save_significant_paths(significant_paths: list[set[int]], filename: str) -> None:
    """
    Saves the significant paths to a file using pickle.

    Parameters:
    - significant_paths (list[set[int]]): The list of significant paths to be saved.
    - filename (str): The name of the file where the paths will be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(significant_paths, file)
    print(f'Significant paths saved to {filename}')

def load_significant_paths(filename: str) -> list[set[int]]:
    """
    Loads the significant paths from a file using pickle.

    Parameters:
    - filename (str): The name of the file from which the paths will be loaded.

    Returns:
    - significant_paths (list[set[int]]): The list of significant paths loaded from the file.
    """
    with open(filename, 'rb') as file:
        significant_paths = pickle.load(file)
    print(f'Significant paths loaded from {filename}')
    return significant_paths

def main(graph_file):
    if os.path.exists(graph_file):
        print(f"Loading graph from {graph_file}...")
        graph = load_graph(graph_file)
    else:
        print(f"Couldn't find graph {graph_file}! Exiting!")
        exit()

    significant_paths = find_path_like_structures(graph, num_walks=10,
                                                    walk_length=end_layer-start_layer,
                                                    start_layer=start_layer,
                                                    end_layer=end_layer)

    save_significant_paths(significant_paths, "paths/pearson_signficant_paths.pkl")
    print(f"len(significant_paths) = {len(significant_paths)}")
    print(f"len(significant_paths[0]) = {len(significant_paths[0])}")
    print(f"len(significant_paths[1]) = {len(significant_paths[1])}")

if __name__ == "__main__":
    graph_file = 'graphs/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.pkl'
    start_layer = 0
    end_layer = 11

    main(graph_file)


# Example usage:
# Assume we have a multipartite graph where each node has a 'layer' attribute
# G = nx.Graph()
# G.add_node(1, layer=1)
# G.add_node(2, layer=2)
# G.add_node(3, layer=3)
# G.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 0.8), (1, 3, 0.6), ...])

# Define the layers you are interested in
# start_layer = 1
# end_layer = 3

# Run the path-like structure detection considering only paths starting in layer 1 and ending in layer 3
# significant_paths = find_path_like_structures(G, num_walks=1000, walk_length=10, start_layer=start_layer, end_layer=end_layer)
# for path in significant_paths:
#     print(path)
