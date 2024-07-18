# %%
import torch as t
import torch
import numpy as np
# %%
# load all of the pearson data in
num_layers = 2
all_corrs_list = []
for first_layer_idx in range(num_layers): # TODO: extent
    # filename = f"artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_correlation_pearson_{first_layer_idx}_{first_layer_idx+1}_1M.1.npz"
    filename = "artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_correlation_pearson_0_1_1M_0.1.npz"
    with open(filename, 'rb') as data:
        all_corrs_list.append(np.load(data)['arr_0'])
all_corrs = np.stack(all_corrs_list, axis=0)
# %%
print(all_corrs.shape)
# %%
# Find the maximum values across the first dimension for each index in the second and third dimensions
max_values = np.max(all_corrs, axis=(1, 2))

# Find the indices of these maximum values
max_indices = np.argmax(all_corrs.reshape(, -1), axis=1)

# Convert flat indices to corresponding (2nd, 3rd) dimension indices
positions = [np.unravel_index(index, all_corrs.shape[1:]) for index in max_indices]

# Convert positions to a (3, 2) array
positions_array = np.array(positions)

print("Max values across the first dimension for each index in the second and third dimensions:\n", max_values)
print("Positions in the 2nd and 3rd dimensions:\n", positions_array)
# %%
# %%
