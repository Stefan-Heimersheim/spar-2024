# %%
# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data


# %%
# Load raw (unclamped) files
folder = '../../artefacts/similarity_measures/mutual_information'
files = [f'{folder}/.unclamped/res_jb_sae_feature_similarity_mutual_information_1M_0.0_{layer}.npz' for layer in range(11)]
matrix = load_similarity_data(files)


# %%
# Count nan, zero and non-zero
number_of_entries = matrix.size
nan_count = np.count_nonzero(np.isnan(matrix))
non_zero_count = np.count_nonzero(matrix) - nan_count
zero_count = number_of_entries - nan_count - non_zero_count

print(f'{number_of_entries=:,}\n{nan_count=:,}\n{zero_count=:,}\n{non_zero_count=:,}')


# %% 
# Create histogram of non-nans
flat_matrix = matrix.flatten()

plt.hist(flat_matrix, bins=200, log=True)
plt.title('Mutual information: Histogram of all SAE feature pairs')
plt.xlabel('Mutual information')
plt.ylabel('Number of SAE feature pairs')
