# %%
# Imports
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data, clamp_low_values, save_compressed


# %%
# Load files and replace nans by zeros
measure_name = 'pearson_correlation'
sae_name = 'openai'
token_desc = '1M'
activity_threshold = 0.0
n_layers = 12

folder = f'../../artefacts/similarity_measures/{measure_name}'
files = [f'{folder}/.unclamped/{sae_name}_sae_feature_similarity_{measure_name}_{token_desc}_{activity_threshold:.1f}_{layer}.npz' for layer in range(n_layers - 1)]
matrix = load_similarity_data(files)
matrix = np.nan_to_num(matrix)

# %%
# Clamp close-to-zero values
clamping_threshold = 0.1
clamp_low_values(matrix, clamping_threshold)

# %%
# Save clamped matrix to file
save_compressed(matrix, f'{folder}/{sae_name}_sae_feature_similarity_{measure_name}_{token_desc}_{activity_threshold:.1f}_{clamping_threshold:.1f}')


# %%
np.count_nonzero(matrix)