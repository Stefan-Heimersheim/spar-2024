# %%
# Imports
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_correlation_data, clamp_low_values, save_compressed


# %%
# Load necessity files, clamp them and save as one file
folder = '../../artefacts/similarity_measures/necessity'
files = [f'{folder}/.unclamped/res_jb_sae_feature_similarity_necessity_{layer}_{layer+1}_1M_0.0.npz' for layer in range(11)]
matrix = load_correlation_data(files)
matrix = np.nan_to_num(matrix)

# %%
clamping_threshold = 0.2
clamp_low_values(matrix, clamping_threshold)

# %%
save_compressed(matrix, f'{folder}/res_jb_sae_feature_similarity_necessity_1M_0.0_{clamping_threshold}')

# %%
