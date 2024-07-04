# %%
import os
# OPTIONAL: Set environment variable to control visibility of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import einops
import numpy as np

from src.pipeline import load_model_and_saes, load_data, run_with_aggregator


# %%
# OPTIONAL: Check if the correct GPU is visible
print(torch.cuda.device_count())  # Should print 1
print(torch.cuda.current_device())  # Should print 0 since it's the first visible device
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)

tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=256)

# %%
class PearsonAggregator:
    def __init__(self, layer, n_features_1, n_features_2):
        """Calculates the pair-wise Pearson correlation of two tensors that are
        provided batch-wise. All computations are done element-wise with broadcasting
        or einsum.

        Args:
            shape (Size): Shape of the result.
        """
        self.layer_1, self.layer_2 = layer, layer + 1

        self.count = 0

        self.sums_1 = torch.zeros(n_features_1)
        self.sums_2 = torch.zeros(n_features_2)

        self.sums_of_squares_1 = torch.zeros(n_features_1)
        self.sums_of_squares_2 = torch.zeros(n_features_2)

        self.sums_1_2 = torch.zeros(n_features_1, n_features_2)

    def process(self, sae_activations):
        layer_1_activations = sae_activations[self.layer_1]
        layer_2_activations = sae_activations[self.layer_2]

        self.count += layer_1_activations.shape[-1]

        self.sums_1 += layer_1_activations.sum(dim=-1)
        self.sums_2 += layer_2_activations.sum(dim=-1)

        self.sums_of_squares_1 += (layer_1_activations ** 2).sum(dim=-1)
        self.sums_of_squares_2 += (layer_2_activations ** 2).sum(dim=-1)

        self.sums_1_2 += einops.einsum(layer_1_activations, layer_2_activations, "f1 t, f2 t -> f1 f2")

    def finalize(self):
        means_1 = self.sums_1 / self.count
        means_2 = self.sums_2 / self.count

        # Compute the covariance, variances, and standard deviations
        covariances = (self.sums_1_2 / self.count) - einops.einsum(
            means_1, means_2, "f1, f2 -> f1 f2"
        )

        variances_1 = (self.sums_of_squares_1 / self.count) - (means_1**2)
        variances_2 = (self.sums_of_squares_2 / self.count) - (means_2**2)

        stds_1 = torch.sqrt(variances_1).unsqueeze(1)
        stds_2 = torch.sqrt(variances_2).unsqueeze(0)

        # Compute the Pearson correlation coefficients
        correlations = covariances / stds_1 / stds_2

        return correlations


# %%
# For each pair of layers, call run_with_aggregator and save the result
output_folder = 'pearson'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

threshold = 0.1
output_filename_fn = lambda layer: f'{output_folder}/res_jb_sae_feature_correlation_pearson_{layer}_{layer+1}_1M_{threshold}.npz'

d_sae = saes[0].cfg.d_sae

for layer in [10]: # range(model.cfg.n_layers - 1):
    aggregator = PearsonAggregator(layer, d_sae, d_sae)

    pearson_correlations = run_with_aggregator(model, saes, 'hook_resid_pre', tokens, aggregator)

    pearson_correlations[pearson_correlations.abs() < threshold] = 0
    pearson_correlations = pearson_correlations.nan_to_num()

    np.savez_compressed(output_filename_fn(layer), pearson_correlations)
