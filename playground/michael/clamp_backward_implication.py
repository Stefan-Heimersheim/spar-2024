# %%
# Imports
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_correlation_data, clamp_low_values, save_compressed


# %%
# Load forward implication files, clamp them and save as one file
folder = '../../artefacts/similarity_measures/backward_implication'
files = [f'{folder}/.unclamped/res_jb_sae_feature_similarity_backward_implication_{layer}_{layer+1}_1M_0.0.npz' for layer in range(11)]
matrix = load_correlation_data(files)
matrix = np.nan_to_num(matrix)

# %%
clamping_threshold = 0.1
clamp_low_values(matrix, clamping_threshold)

# %%
save_compressed(matrix, f'{folder}/res_jb_sae_feature_similarity_backward_implication_1M_0.0_{clamping_threshold}')

# %%
