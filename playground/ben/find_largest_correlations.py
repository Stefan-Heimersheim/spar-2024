# %%
import torch as t
import torch
import numpy as np
# %%
# load all of the pearson data
filename = f"artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz"
with open(filename, 'rb') as data:
    interaction_data = np.load(data)['arr_0']
# %%
def ranked_pairs(matrix: np.ndarray) -> np.ndarray:
    m, _ = matrix.shape
    
    # Step 1: Flatten the matrix (shape: [m*m])
    flattened_matrix = matrix.flatten()
    
    # Step 2: Create row and column indices
    row_indices = np.repeat(np.arange(m), m)
    col_indices = np.tile(np.arange(m), m)
    
    # Step 3: Create the final array with indices and values
    values = flattened_matrix
    result = np.column_stack((row_indices, col_indices, values))
    
    # Step 4: Sort the result array by values
    sorted_result = result[result[:, 2].argsort()[::-1]]
    
    return sorted_result
# %%
num_layers, d_sae, _ = interaction_data.shape
"""
1. run a single row through.
it activates L0f{4, 800, 12000} > 0 and that's it

"""

# %%

# %%
# Find the indices of these maximum values
# TODO: scale this up, remove truncation
flattened_pairs = interaction_data.reshape(num_layers, -1)[:,:10000000]
max_indices = np.argsort(flattened_pairs, axis=1)[:, ::-1]
# Convert flat indices to corresponding (2nd, 3rd) dimension indices
positions_array = np.array(np.unravel_index(max_indices, interaction_data.shape[1:]))

# print("Max values across the first dimension for each index in the second and third dimensions:\n", max_values)
print("Positions in the 2nd and 3rd dimensions:\n", positions_array)
# %%
# all_corrs[0, 14525, 11914]
# %%
