# %%
import torch as t
import numpy as np
import glob
import os
import re
from collections import defaultdict

if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"
d_sae = 24576
print(f"Loaded {device=}, {d_sae=}")
# %%
def load_tensors_into_dict(directory):
    # Dictionary to store the loaded tensors
    tensor_dict = {}
    
    # Pattern to match the filenames
    pattern = r'layer_(\d+)__feat_(\d+)__num_batches_\d+__batch_size_\d+\.pth'
    count = 0
    # Find all .pth files in the directory
    for filepath in glob.glob(os.path.join(directory, '*.pth')):
        # Extract layer_idx and feat_idx from the filename
        match = re.search(pattern, os.path.basename(filepath))
        if match:
            layer_idx, feat_idx = match.groups()
            
            # Create the key
            key = f'layer_{layer_idx}_feat_{feat_idx}'
            
            # Load the tensor
            loaded_tensor = t.load(filepath, map_location=t.device(device))
            
 
            matrixes_to_add = ['masked_means', 'masked_mse', 'mean_diffs', 'mse']
            # Add to the dictionary
            tensor_dict[key] = {
                attr: loaded_tensor[attr]
                for attr in matrixes_to_add
            }
        count += 1
        if count > 25:
            break
    
    return tensor_dict


# %%
directory = 'artefacts/ablations'
result = load_tensors_into_dict(directory)
# %%
pearson_corr_filename = f"artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz"
with open(pearson_corr_filename, 'rb') as data:
    interaction_data = np.load(data)['arr_0']
# %%
top_corr_flat_idx = np.load("artefacts/sampled_interaction_measures/pearson_correlation/count_1000.npz")['arr_0']
# for all high-value pearson correlations, loading the first feature from each pair in a (num_layers, num_top_features) matrix
corr_prev_layer_feat_idxes, corr_next_layer_feat_idxes = (
    np.array([
        np.unravel_index(top_corr_flat_idx[layer_idx], shape=(d_sae, d_sae))[ordering_idx]
        for layer_idx in range(top_corr_flat_idx.shape[0])
    ])
    for ordering_idx in range(2)
)
# %%
# TODO: only load the parts that you want from the files, rather than all of them. do it for the first layer
def parse_key(key):
    match = re.match(r"layer_(\d+)_feat_(\d+)", key)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None
# %%
