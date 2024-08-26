# %%
import os
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from similarity_helpers import get_filename, load_similarity_data, clamp_low_values, save_compressed


# %%
measure_name = "pearson_correlation"
sae_name = 'res_jb_sae'
n_layers = 12
activation_threshold = 0.2
n_tokens = '100M'

folder = f'../../../artefacts/similarity_measures/{measure_name}/.unclamped'
files = [f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens=n_tokens, first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)]

matrix = load_similarity_data(files)
matrix = np.nan_to_num(matrix)
np.count_nonzero(matrix)

# %%
clamping_threshold = 0.1
clamp_low_values(matrix, clamping_threshold)
np.count_nonzero(matrix)


# %%
save_compressed(matrix, f'../../../artefacts/similarity_measures/{measure_name}/{get_filename(measure_name, "feature_similarity", activation_threshold, clamping_threshold, n_tokens=n_tokens, sae_name=sae_name)}')
