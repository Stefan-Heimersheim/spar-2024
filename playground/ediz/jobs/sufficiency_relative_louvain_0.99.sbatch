#!/bin/bash

#SBATCH --job-name=sufficiency_relative_louvain_0_99        # Job name
#SBATCH --output=jobs/logs/%x_%j.out           # Standard output and error log
#SBATCH --error=jobs/logs/%x_%j.err            # Standard error log

#SBATCH --partition=single                     # Partition name

# Activate the Python virtual environment
source ../../.venv/bin/activate

# Define your parameters

# Required: Path to the .npz file containing the graph data
npz_file="../../artefacts/similarity_measures/sufficiency_relative_activation/res_jb_sae_feature_similarity_sufficiency_relative_activation_10M_0.2_0.1.npz"

# Required: Path to the file where the graph will be saved/loaded
graph_file="graphs/res_jb_sae_feature_similarity_sufficiency_10M_relative_activation_0.2_threshold_0.99.pkl"

# Required: File name for output files
file_name="sufficiency_louvain_threshold_0.99_plotly"

# Optional: Minimum edge weight to include in the graph
threshold=0.99

# Optional: Partition method ('louvain' or 'leiden')
partition_method="louvain"

# Optional: Quality function ('modularity', 'cpm', 'rb_pots', etc.)
# quality_function="modularity"

# Optional: Use weighted partitioning (true or false)
weighted_partition=false

# Optional: Path to the mask file
# mask_file="path/to/mask_file.npz"

# Optional: Resolution parameter for Leiden algorithm
# resolution_parameter=0.1

# Construct the base command with required arguments
cmd="python3 -u partition_pipeline.py --npz_file=$npz_file --graph_file=$graph_file --file_name=$file_name"

# Append optional parameters only if they are defined
[ ! -z "$threshold" ] && cmd+=" --threshold=$threshold"
[ ! -z "$partition_method" ] && cmd+=" --partition_method=$partition_method"
[ ! -z "$quality_function" ] && cmd+=" --quality_function=$quality_function"
[ ! -z "$weighted_partition" ] && cmd+=" --weighted_partition=$weighted_partition"
[ ! -z "$mask_file" ] && cmd+=" --mask_file=$mask_file"
[ ! -z "$resolution_parameter" ] && cmd+=" --resolution_parameter=$resolution_parameter"

# Print the command (for debugging purposes)
echo "Running command: $cmd"
# Execute the command
$cmd
