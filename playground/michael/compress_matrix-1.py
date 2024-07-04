# %%
import torch
import numpy as np


# %%
def convert_and_compress(input_filename_fn, output_filename_fn, n_layers, threshold):
    for layer in range(n_layers - 1):
        print(f'Layer {layer}:')
        with open(input_filename_fn(layer), 'rb') as f:
            correlations = torch.load(f)

        correlations[correlations.abs() < threshold] = 0
        correlations = correlations.nan_to_num()

        print(f'There are {correlations.count_nonzero()} entries above the threshold.')

        np.savez_compressed(output_filename_fn(layer), correlations)
        print()


# %%
threshold = 0.1
input_filename_fn = lambda layer: f'pearson/res_jb_sae_feature_correlation_pearson_{layer}_{layer+1}.pt'
output_filename_fn = lambda layer: f'clamped_pearson/res_jb_sae_feature_correlation_pearson_{layer}_{layer+1}_{threshold}.npz'

convert_and_compress(input_filename_fn, output_filename_fn, n_layers=12, threshold=threshold)