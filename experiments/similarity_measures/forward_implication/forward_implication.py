# %%
import os
# OPTIONAL: Set environment variable to control visibility of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import einops
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data, run_with_aggregator
from similarity_measures import ForwardImplicationAggregator


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
number_of_batches, number_of_token_desc = 256, '1M'
# number_of_batches, number_of_token_desc = 4269, '17.5M'


# %%
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)

tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=number_of_batches)


# %%
# For each pair of layers, call run_with_aggregator and save the result
measure_name = 'forward_implication'

output_folder = f'../../../artefacts/similarity_measures/{measure_name}/.unclamped'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

activity_lower_bound = 0.0
output_filename_fn = lambda layer: f'{output_folder}/res_jb_sae_feature_similarity_{measure_name}_{layer}_{layer+1}_{number_of_token_desc}_{activity_lower_bound}.npz'

d_sae = saes[0].cfg.d_sae

for layer in range(model.cfg.n_layers - 1):
    aggregator = ForwardImplicationAggregator(layer, (d_sae, d_sae), lower_bound=activity_lower_bound)

    forward_implications = run_with_aggregator(model, saes, 'hook_resid_pre', tokens, aggregator)

    np.savez_compressed(output_filename_fn(layer), forward_implications)
