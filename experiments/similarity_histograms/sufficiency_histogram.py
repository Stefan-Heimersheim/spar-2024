# %%
# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data, get_filename


# %%
# Load raw (unclamped) files
measure_name = "sufficiency_relative_activation"
sae_name = 'res_jb_sae'
n_layers = 12
activation_threshold = 0.2

folder = f'../../artefacts/similarity_measures/{measure_name}/.unclamped'
matrix = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens="10M", first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])



# %%
# Count nan, zero and non-zero
number_of_entries = matrix.size
nan_count = np.count_nonzero(np.isnan(matrix))
non_zero_count = np.count_nonzero(matrix) - nan_count
zero_count = number_of_entries - nan_count - non_zero_count


# %%
print(f'{number_of_entries=:,}\n{nan_count=:,}\n{zero_count=:,}\n{non_zero_count=:,}')


# %%
# Create histogram of non-nans
flat_matrix = matrix.flatten()

plt.hist(flat_matrix, bins=200, log=True)
plt.title('Sufficiency: Histogram of all SAE feature pairs')
plt.xlabel('Sufficiency')
plt.ylabel('Number of SAE feature pairs')
