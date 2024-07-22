# %%
import torch as t
import torch
import numpy as np
# %%
# load all of the jaccard data
num_layers = 2
all_corrs_list = []
# for first_layer_idx in range(num_layers): # TODO: extent
    # filename = f"artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_correlation_pearson_{first_layer_idx}_{first_layer_idx+1}_1M.1.npz"
filename = f"artefacts/similarity_measures/jaccard_similarity/res_jb_sae_feature_similarity_jaccard_similarity_1M_0.0_0.1.npz"
with open(filename, 'rb') as data:
    jaccard_data = np.load(data)['arr_0']
# all_corrs = np.stack(all_corrs_list, axis=0)
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
def find_random_positions(matrix: np.ndarray, min_val=0.5):
    # Find all positions where the value is greater than 0.4
    all_idxes = np.argwhere(matrix > 0.4)

    # Randomly select 100 positions from these
    random_indices = np.random.choice(all_idxes.shape[0], 100, replace=False)
    return all_idxes[random_indices]

# %%
# naive greedy graph-making algorithm
def create_graph(
    interactions: np.ndarray, start_layer_idx, start_layer_feat_idx, next_layer_feat_idx,
):
    curr_layer_idx = start_layer_idx
    prev_nodes = []
    while curr_layer_idx > 0: # TODO: >= or > ?
        pass
    


# Find the maximum values across the first dimension for each index in the second and third dimensions
max_values = np.max(all_corrs, axis=(1, 2))

# Find the indices of these maximum values
max_indices = np.argmax(all_corrs.reshape(num_layers, -1), axis=1)

# Convert flat indices to corresponding (2nd, 3rd) dimension indices
positions = [np.unravel_index(index, all_corrs.shape[1:]) for index in max_indices]

# Convert positions to a (3, 2) array
positions_array = np.array(positions)

print("Max values across the first dimension for each index in the second and third dimensions:\n", max_values)
print("Positions in the 2nd and 3rd dimensions:\n", positions_array)
# %%
all_corrs[0, 14525, 11914]
# %%
