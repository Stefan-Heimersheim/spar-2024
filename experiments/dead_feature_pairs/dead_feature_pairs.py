# Find out how many tokens are needed to get a good representation of
# SAE feature activations
#
# Idea:
# - SAE activations are sparse, so we need lots of tokens to get a 
#   sufficient number of non-zero activations for any given feature
# - This is exacerbated by the fact that we're looking at feature pairs:
#   For each pair, we need a sufficient number of tokens where _both_
#   features are active
# - To analyze how many tokens we need to run our experiments on, we
#   simply count the size of the "activation overlap" for all pairs of 
#   feature from adjacent layers
# - We do this for different amounts of tokens and see where the number of
#   empty intersections is saturated
# - This should give us a good feeling for the number of tokens required
#   for more involved experiments


# %%
import os
# OPTIONAL: Set environment variable to control visibility of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import einops
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data


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
# Define number of tokens
# number_of_batches, number_of_token_desc = 1, '4k'
# number_of_batches, number_of_token_desc = 32, '128k'
# number_of_batches, number_of_token_desc = 256, '1M'
# number_of_batches, number_of_token_desc = 1024, '4M'
number_of_batches, number_of_token_desc = 4269, '17.5M'


# %%
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)

tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=number_of_batches)


# %%
class DeadFeaturePairsAggregator:
    def __init__(self, first_layer, n_features_1, n_features_2, lower_bound=0.0):
        """Calculates the pair-wise Pearson correlation of two tensors that are
        provided batch-wise. All computations are done element-wise with broadcasting
        or einsum.

        Args:
            shape (Size): Shape of the result.
        """
        self.layer_1, self.layer_2 = first_layer, first_layer + 1
        self.lower_bound = lower_bound

        self.sums = torch.zeros(n_features_1, n_features_2, dtype=torch.int)

    def process(self, sae_activations):
        active_1 = (sae_activations[self.layer_1] > self.lower_bound).float()
        active_2 = (sae_activations[self.layer_2] > self.lower_bound).float()

        self.sums += einops.einsum(active_1, active_2, 'f1 t, f2 t -> f1 f2').int()

    def finalize(self):
        return self.sums


# %%
# For each pair of layers, run model with aggregator and save intermediate results
measure_name = 'dead_feature_pairs'
batch_size = 32
hook_name = 'hook_resid_pre'

output_folder = measure_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

activity_lower_bound = 0.0
overlap_thresholds = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000]
evaluation_frequency = 16  # Evaluate number of empty-overlap pairs every X batches
output_filename_fn = lambda layer: f'{output_folder}/res_jb_sae_{measure_name}_{layer}_{layer+1}_{number_of_token_desc}_{activity_lower_bound}.npz'

d_sae = saes[0].cfg.d_sae

for layer in range(model.cfg.n_layers - 1):
    aggregator = DeadFeaturePairsAggregator(layer, d_sae, d_sae, lower_bound=activity_lower_bound)

    data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

    context_size = saes[0].cfg.context_size
    d_sae = saes[0].cfg.d_sae
    sae_activations = torch.empty(model.cfg.n_layers, d_sae, batch_size * context_size)


    def retrieval_hook(activations, hook):
        layer = hook.layer()

        sae_activations[layer] = einops.rearrange(
            saes[layer].encode(activations), "batch seq features -> features (batch seq)"
        )


    model.add_hook(lambda name: name.endswith(f".{hook_name}"), retrieval_hook)

    evaluation_matrix = []
    with torch.no_grad():
        for index, batch_tokens in tqdm(enumerate(data_loader), total=len(data_loader)):
            model.run_with_hooks(batch_tokens)

            # Now we can use sae_activations
            aggregator.process(sae_activations)

            # At certain points, evaluate the current overlap
            if (index + 1) % evaluation_frequency == 0:
                evaluation_matrix.append([(aggregator.sums <= threshold).sum() for threshold in overlap_thresholds])

        # Save overlap data
        np.savez_compressed(output_filename_fn(layer), np.array(evaluation_matrix).T)


# %%
# Load and plot stats
output_folder = '../../experiments/activation_overlap_over_time'
overlap_thresholds = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000]
evaluation_frequency = 16 * 32 * 128
layer = 1
d_sae = 24576

evaluation_matrix = np.load(f'{output_folder}/res_jb_sae_feature_correlation_activation_overlap_over_time_{layer}_{layer+1}_4M_0.0.npz')['arr_0']
evaluation_matrix = np.concatenate([np.ones((len(overlap_thresholds), 1)) * d_sae * d_sae, evaluation_matrix], axis=-1)

for index, threshold in enumerate(overlap_thresholds):
    plt.plot(np.arange(len(evaluation_matrix[index])) * evaluation_frequency, evaluation_matrix[index], label=f'X = {threshold}')

plt.legend()
plt.xlabel('Number of tokens')
plt.ylabel('Number of feature pairs with X or less co-activations')
plt.title(f'Dead SAE feature pairs in layers {layer} and {layer+1}')
plt.show()
