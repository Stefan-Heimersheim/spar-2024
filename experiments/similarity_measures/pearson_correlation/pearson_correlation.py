# %%
import os
# OPTIONAL: Set environment variable to control visibility of GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data, run_with_aggregator
from similarity_measures import PearsonCorrelationAggregator


# %%
# OPTIONAL: Check if the correct GPU is visible
# print(torch.cuda.device_count())  # Should print 1
# print(torch.cuda.current_device())  # Should print 0 since it's the first visible device
# print(torch.cuda.get_device_name(0))  # Should print the name of the GPU

# if torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"Device: {device}")
device = 'cuda:0'


# %%
# Define number of tokens
number_of_batches, number_of_token_desc = 1, '4k'
# number_of_batches, number_of_token_desc = 32, '128k'
# number_of_batches, number_of_token_desc = 256, '1M'
# number_of_batches, number_of_token_desc = 2560, '10M'
# number_of_batches, number_of_token_desc = 4269, '17.5M'


# %%
model_name = 'gemma-2-2b'
sae_name = 'gemma-scope-2b-pt-res-canonical'
hook_name = 'hook_resid_pre' if model_name == 'gpt2-small' else 'width_16k/canonical'
model, saes = load_model_and_saes(model_name=model_name, sae_name=sae_name, hook_name=hook_name, device=device)
tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=number_of_batches)
print("Loaded model and tokens")

# %%
# For each pair of layers, call run_with_aggregator and save the result
measure_name = 'pearson_correlation'

output_folder = f'../../../artefacts/similarity_measures/{model_name}/{measure_name}/.unclamped'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_filename_fn = lambda layer: f'{output_folder}/res_jb_sae_feature_similarity_{measure_name}_{number_of_token_desc}_{layer}.npz'

d_sae = saes[0].cfg.d_sae

print(f"Calculating {measure_name}...")
for layer in range(model.cfg.n_layers - 1):
    aggregator = PearsonCorrelationAggregator(layer, (d_sae, d_sae))

    pearson_correlations = run_with_aggregator(model, saes, hook_name, tokens, aggregator, device)

    np.savez_compressed(output_filename_fn(layer), pearson_correlations)
print(f"Finished calculating {measure_name}")