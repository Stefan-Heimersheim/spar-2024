import argparse
from ediz_6 import main  # Replace 'your_module' with the actual module name where main is defined
import leidenalg as la  # Ensure this is imported if 'la.CPMVertexPartition' is used

# Create a lookup dictionary for quality functions
quality_function_map = {
    "modularity": la.ModularityVertexPartition,          # Maximizes modularity
    "cpm": la.CPMVertexPartition,                        # Constant Potts Model (CPM)
    "rb_pots": la.RBConfigurationVertexPartition,        # Reichardt and Bornholdt's configuration model
    "rber_pots": la.RBERVertexPartition,                 # Reichardt and Bornholdt's Erdős–Rényi model
    "significance": la.SignificanceVertexPartition,      # Maximizes significance
    "surprise": la.SurpriseVertexPartition,              # Maximizes surprise
}

def run():
    print("Parsing arguments!")
    parser = argparse.ArgumentParser(description="Run graph partitioning.")
    
    # Required arguments
    parser.add_argument("--npz_file", required=True, help="Path to the .npz file containing the graph data.")
    parser.add_argument("--graph_file", required=True, help="Path to the file where the graph will be saved/loaded.")
    parser.add_argument("--file_name", required=True, help="File name for output.")

    # Optional arguments with defaults
    parser.add_argument("--mask_file", default=None, help="Path to the mask file. Optional.")
    parser.add_argument("--partition_method", default="louvain", choices=["louvain", "leiden"], help="Partition method to use. Default is 'louvain'.")
    parser.add_argument("--resolution_parameter", type=float, default=None, help="Resolution parameter for Leiden. Optional.")
    parser.add_argument("--quality_function", default="modularity", choices=quality_function_map.keys(), help="Quality function to use in Leiden partition. Default is ModularityVertexPartition.")
    parser.add_argument("--weighted_partition", type=bool, default=True, help="Use weighted partitioning. Default is True.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum edge weight to include in the graph. Default is 0.0.")

    args = parser.parse_args()

    # Map the quality_function argument to the actual function using the lookup dictionary
    quality_function = quality_function_map[args.quality_function]
    print("Complete.\nEntering main function!")
    # Call the original main function with parsed arguments
    main(
        npz_file=args.npz_file,
        graph_file=args.graph_file,
        file_name=args.file_name,
        mask_file=args.mask_file,
        partition_method=args.partition_method,
        resolution_parameter=args.resolution_parameter,
        quality_function=quality_function,
        weighted_partition=args.weighted_partition,
        threshold=args.threshold
    )




if __name__ == "__main__":
    run()


