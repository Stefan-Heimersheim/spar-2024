# %%
# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_correlation_data


# %%
# Load raw (unclamped) files
folder = '../../artefacts/similarity_measures/pearson_correlation'
files = [f'{folder}/.unclamped/res_jb_sae_feature_similarity_pearson_correlation_{layer}_{layer+1}_1M_0.0.npz' for layer in range(11)]
matrix = load_correlation_data(files)


# %%
# Count nan, zero and non-zero
number_of_entries = matrix.size
nan_count = np.count_nonzero(np.isnan(matrix))
non_zero_count = np.count_nonzero(matrix) - nan_count
zero_count = number_of_entries - nan_count - non_zero_count


# %%
print(f'{number_of_entries=:,}\n{nan_count=:,}\n{zero_count=:,}\n{non_zero_count=:,}')


# %% Create histogram of non-nans/non-zeros
flat_matrix = matrix.flatten()
nonzero_values = flat_matrix[flat_matrix > 0]

plt.hist(nonzero_values, bins=100, log=True)
plt.title('Pearson correlation: Histogram of all SAE feature pairs')
plt.xlabel('Pearson correlation')
plt.ylabel('Number of SAE feature pairs')
