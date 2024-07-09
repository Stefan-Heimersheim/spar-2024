# Full experimental pipeline
#
# This file contains all steps required to reproduce the results
# of the _SAE feature interaction graph_ SPAR project.
#
# The main steps are:
# 1. Create feature similarity matrices for all considered measures.
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
