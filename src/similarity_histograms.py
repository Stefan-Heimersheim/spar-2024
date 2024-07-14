# %%
# Imports
from typing import Optional, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from similarity_helpers import load_correlation_data


def create_similarity_histogram(
        similarity_measure: str, 
        n_layers: int = 12, 
        data_folder: Optional[str] = None, 
        filename_fn: Optional[Callable] = None,
        similarity_measure_str: Optional[str] = None
    ) -> Tuple[int, int, int, int]:
    # If a filename_fn is given, just use it
    if filename_fn is None:
        # Otherwise, check if a folder is given
        if data_folder is None:
            # If not, use the standard artefacts folder
            data_folder = f'../../artefacts/similarity_measures/{similarity_measure}/.unclamped'

        # Using the folder, build a filename_fn
        filename_fn = lambda layer: f'{data_folder}/res_jb_sae_feature_similarity_{similarity_measure}_{layer}_{layer+1}_1M_0.0.npz'

    # Load raw (unclamped) files for all layer pairs
    files = [filename_fn(layer) for layer in range(n_layers - 1)]
    matrix = load_correlation_data(files)

    # Count nan, zero and non-zero
    number_of_entries = matrix.size
    nan_count = np.count_nonzero(np.isnan(matrix))
    non_zero_count = np.count_nonzero(matrix) - nan_count
    zero_count = number_of_entries - nan_count - non_zero_count

    # Create histogram of non-nans/non-zeros
    if similarity_measure_str is None:
        similarity_measure_str = similarity_measure

    flat_matrix = matrix.flatten()
    nonzero_values = flat_matrix[flat_matrix > 0]

    plt.hist(nonzero_values, bins=100, log=True)
    plt.title(f'{similarity_measure_str}: Histogram of all SAE feature pairs')
    plt.xlabel(f'{similarity_measure_str}')
    plt.ylabel('Number of SAE feature pairs')

    return number_of_entries, nan_count, zero_count, non_zero_count
