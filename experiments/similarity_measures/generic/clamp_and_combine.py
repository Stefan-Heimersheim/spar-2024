# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import einops
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from similarity_helpers import get_filename, load_similarity_data, clamp_low_values, save_compressed


# %%
print(torch.cuda.device_count())  # Should print 1
print(torch.cuda.current_device())  # Should print 0 since it's the first visible device
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
measure_name = "jaccard_similarity_relative_activation"
sae_name = 'res_jb_sae'
n_layers = 12
activation_threshold = 0.3

folder = f'../../../artefacts/similarity_measures/{measure_name}/.unclamped'
files = [f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens="1M", first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)]

matrix = load_similarity_data(files)
matrix = np.nan_to_num(matrix)

# %%
clamping_threshold = 0.1
clamp_low_values(matrix, clamping_threshold)
np.count_nonzero(matrix)

# %%
save_compressed(matrix, f'../../../artefacts/similarity_measures/{measure_name}/{get_filename(measure_name, "feature_similarity", None, clamping_threshold, n_tokens="1M", sae_name=sae_name)}')
