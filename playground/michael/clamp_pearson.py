# %%
# Imports
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_correlation_data, clamp_low_values, save_compressed


# %%
# Load Pearson files and replace nans by zeros
folder = '../../artefacts/similarity_measures/pearson_correlation'
files = [f'{folder}/.unclamped/res_jb_sae_feature_similarity_pearson_correlation_{layer}_{layer+1}_1M_0.0.npz' for layer in range(11)]
matrix = load_correlation_data(files)
matrix = np.nan_to_num(matrix)

# %%
# Clamp close-to-zero values
clamping_threshold = 0.1
clamp_low_values(matrix, clamping_threshold)

# %%
# Save clamped matrix to file
save_compressed(matrix, f'{folder}/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_{clamping_threshold}')


# %%
np.count_nonzero(matrix)