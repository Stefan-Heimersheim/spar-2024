# Full experimental pipeline
#
# This file contains all steps required to reproduce the results
# of the _SAE feature interaction graph_ SPAR project.
#
# The main steps are:
# 1. Create feature similarity matrices for all considered measures.
#    1.1 Create and locally save full matrices (one per layer pair)
#    1.2 Create stats (e.g., histograms) from the raw data
#    1.3 Compress matrices by clamping close-to-zero values
#    1.4 Store the compressed matrices (one per measure) in /artefacts
# 2. Build a feature similarity graph from each matrix.
# 3. Identify feature pairs for causal analysis.
# 4. Build a causal similarity graph via activation patching.
# 5. Analyze the graph's structure.


# %%
# Imports


# %%
# Configuration
similarity_measures = ['cosine', 'pearson', 'jaccard', 'necessity', 'sufficiency']


# %%
# Step 1: Create feature similarity matrices for all considered measures.
#
# Details:
# - Use 1M tokens (256 batches * 32 rows * 128 tokens) as the dataset
# - Cut off low activations for binary measures
# - Replace low similarity values with zero to save space
# - Save the similarity matrices to individual files (one per pair of layers)
#   using np.savez_compressed()


# %%
# Step 2: Build a feature similarity graph from each matrix.


# %%
# Step 3: Identify feature pairs for causal analysis.


# %%
# Step 4: Build a causal interaction graph via activation patching.


# %%
# Step 5: Analyze the graph's structure.
