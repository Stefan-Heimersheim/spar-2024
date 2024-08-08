# %%
# Imports
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_correlation_data, clamp_low_values, save_compressed


# %%
# Load MI files, clamp them and save as one file
folder = '../../artefacts/similarity_measures/mutual_information'
files = [f'{folder}/.unclamped/res_jb_sae_feature_correlation_mutual_information_1M_0.0_{layer}.npz' for layer in range(11)]
matrix = load_correlation_data(files)
matrix = np.nan_to_num(matrix)

# %%
clamping_threshold = 0.3
clamp_low_values(matrix, clamping_threshold)

# %%
save_compressed(matrix, f'{folder}/res_jb_sae_feature_similarity_mutual_information_1M_0.0_{clamping_threshold}')
