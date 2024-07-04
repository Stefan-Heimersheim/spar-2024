# %%
import os
# OPTIONAL: Set environment variable to control visibility of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
class ForwardImplicationAggregator:
    def __init__(self, layer, n_features_1, n_features_2, lower_bound=0.0):
        """Calculates the pair-wise Pearson correlation of two tensors that are
        provided batch-wise. All computations are done element-wise with broadcasting
        or einsum.

        Args:
            shape (Size): Shape of the result.
        """
        self.layer_1, self.layer_2 = layer, layer + 1
        self.lower_bound = lower_bound

        self.counts = torch.zeros(n_features_1, n_features_2)
        self.sums = torch.zeros(n_features_1, n_features_2)

    def process(self, sae_activations):
        layer_1_activations = sae_activations[self.layer_1]
        layer_2_activations = sae_activations[self.layer_2]

        active_1 = (layer_1_activations > self.lower_bound).float()
        active_2 = (layer_2_activations > self.lower_bound).float()

        self.counts += active_1.sum(dim=-1).unsqueeze(1)
        self.sums += einops.einsum(active_1, active_2, 'f1 t, f2 t -> f1 f2')

    def finalize(self):
        return self.sums / self.counts


# %%
# For each pair of layers, call run_with_aggregator and save the result
output_folder = 'forward_implication'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

activity_lower_bound = 0.0
interaction_threshold = 0.1
output_filename_fn = lambda layer: f'{output_folder}/res_jb_sae_feature_correlation_forward_implication_{layer}_{layer+1}_1M_{threshold}.npz'

d_sae = saes[0].cfg.d_sae

for layer in range(model.cfg.n_layers - 1):
    aggregator = ForwardImplicationAggregator(layer, d_sae, d_sae, lower_bound=activity_lower_bound)

    forward_implications = run_with_aggregator(model, saes, 'hook_resid_pre', tokens, aggregator)

    forward_implications[forward_implications.abs() < interaction_threshold] = 0
    forward_implications = forward_implications.nan_to_num()

    np.savez_compressed(output_filename_fn(layer), forward_implications)


# %%
# Manual computation as a sanity check
from torch.utils.data import DataLoader

# Nested for loops with own calculation
def mutual_1(tensor_1, tensor_2, lower_bound=0.0):
    active_1 = tensor_1 > lower_bound
    active_2 = tensor_2 > lower_bound

    result = torch.empty(active_1.shape[0], active_2.shape[0])
    for index_1, feature_1 in enumerate(active_1):
        for index_2, feature_2 in enumerate(active_2):
            result[index_1, index_2] = (feature_1 * feature_2).sum() / feature_1.sum()

    return result


# Own calculation with broadcasting
def mutual_2(tensor_1, tensor_2, lower_bound=0.0):
    active_1 = (tensor_1 > lower_bound).unsqueeze(dim=1)
    active_2 = (tensor_2 > lower_bound).unsqueeze(dim=0)

    return (active_1 * active_2).sum(dim=-1) / active_1.sum(dim=-1)


f1 = torch.maximum(torch.rand(10, 1000) - 0.9, torch.tensor([0]))
f2 = torch.maximum(torch.rand(20, 1000) - 0.9, torch.tensor([0]))

true_result = mutual_1(f1, f2)
print(f'True forward implication: {true_result}')

batch_size = 100
loader_1 = DataLoader(f1.movedim(-1, 0), batch_size=batch_size)
loader_2 = DataLoader(f2.movedim(-1, 0), batch_size=batch_size)
aggregator = ForwardImplicationAggregator(0, f1.shape[0], f2.shape[0])

for input_1, input_2 in zip(loader_1, loader_2):
    aggregator.process([input_1.movedim(0, -1), input_2.movedim(0, -1)])

our_result = aggregator.finalize()
print(f'Calculated forward implication: {our_result}')

# our_result = mutual_2(f1, f2)

print(f'{torch.allclose(true_result, our_result, atol=2e-6)=}')
