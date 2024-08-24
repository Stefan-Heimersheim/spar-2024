import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import pickle
import os

def load_graph_from_pickle(file_path):
    """
    Loads a NetworkX graph from a pickle file.
    
    Parameters:
    file_path (str): Path to the pickle file containing the graph.
    
    Returns:
    graph (networkx.Graph): Loaded graph.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
        
    if not isinstance(graph, nx.Graph):
        raise ValueError("The loaded file does not contain a valid NetworkX graph.")
    
    return graph

def compute_laplacian(graph):
    """
    Computes the Laplacian matrix of a graph.
    
    Parameters:
    graph (networkx.Graph): The input graph.
    
    Returns:
    laplacian (np.ndarray): The Laplacian matrix of the graph.
    """
    laplacian = nx.laplacian_matrix(graph).toarray()
    return laplacian

def eigendecomposition(matrix, big = False):
    """
    Computes the eigendecomposition of a matrix.
    
    Parameters:
    matrix (np.ndarray): The matrix to decompose.
    
    Returns:
    eigenvalues (np.ndarray): The eigenvalues of the matrix.
    eigenvectors (np.ndarray): The eigenvectors of the matrix.
    """
    if not big:
        eigenvalues, eigenvectors = eigh(matrix)
    else:
        eigenvalues, eigenvectors = eigsh(matrix, k=6)
    return eigenvalues, eigenvectors

def analyze_laplacian_eigendecomposition(eigenvalues, eigenvectors):
    """
    Analyzes the eigendecomposition of the Laplacian matrix.
    
    Parameters:
    eigenvalues (np.ndarray): The eigenvalues of the Laplacian.
    eigenvectors (np.ndarray): The eigenvectors of the Laplacian.
    
    Returns:
    insights (str): A description of insights derived from the analysis.
    """
    insights = []

    # Smallest eigenvalue (should be close to 0)
    smallest_eigenvalue = np.min(eigenvalues)
    insights.append(f"Smallest eigenvalue: {smallest_eigenvalue} (should be close to 0 for connected components).")

    # Number of zero eigenvalues indicates the number of connected components
    zero_eigenvalues_count = np.sum(eigenvalues < 1e-10)
    insights.append(f"Number of zero eigenvalues: {zero_eigenvalues_count} (indicates {zero_eigenvalues_count} connected component(s)).")

    # Fiedler value and vector (second smallest eigenvalue)
    fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else None
    if fiedler_value is not None:
        insights.append(f"Fiedler value (second smallest eigenvalue): {fiedler_value}.")
        insights.append("Fiedler vector (associated eigenvector) gives insight into graph partitioning.")
    
    return "\n".join(insights)

def main(file_path):
    try:
        print("Starting process...")
        
        # Load the graph
        print(f"Loading graph from {file_path}...")
        graph = load_graph_from_pickle(file_path)
        print("Graph successfully loaded.")
        
        # Compute the Laplacian matrix
        print("Computing the Laplacian matrix...")
        laplacian = compute_laplacian(graph)
        print("Laplacian matrix computed.")
        
        # Perform eigendecomposition
        print("Performing eigendecomposition on the Laplacian matrix...")
        eigenvalues, eigenvectors = eigendecomposition(laplacian, big=True)
        print("Eigendecomposition completed.")
        
        # Analyze and describe the eigendecomposition
        print("Analyzing eigendecomposition results...")
        insights = analyze_laplacian_eigendecomposition(eigenvalues, eigenvectors)
        print("Analysis completed.")
        
        # Display results
        print("\nLaplacian Matrix:")
        print(laplacian)
        print("\nEigenvalues:")
        print(eigenvalues)
        print("\nInsights:")
        print(insights)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = "graphs/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.pkl"
    main(file_path)
