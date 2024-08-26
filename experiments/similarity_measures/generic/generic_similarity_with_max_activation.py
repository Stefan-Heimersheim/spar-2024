# %%
import os
import torch
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data, run_with_aggregator, load_data_from_file
from similarity_helpers import get_aggregator


# %%

device = 'cuda:1'
# Define number of tokens
# number_of_batches, number_of_token_desc = 1, '4k'
# number_of_batches, number_of_token_desc = 32, '128k'
# number_of_batches, number_of_token_desc = 256, '1M'
# number_of_batches, number_of_token_desc = 2560, '10M'
# number_of_batches, number_of_token_desc = 4269, '17.5M'
number_of_batches, number_of_token_desc = 24605, '100M'
    

# %%
print('Loading model and SAEs...')
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)

# tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=number_of_batches).to(device)
tokens = load_data_from_file(model, saes[0], '../../../artefacts/pile_subset.json', number_of_batches=number_of_batches).to(device)


# %%
measure_name = 'pearson_correlation'

# For each pair of layers, call run_with_aggregator and save the result
output_folder = f'../../../artefacts/similarity_measures/{measure_name}/.unclamped'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

max_activations = torch.tensor(np.load('../../../artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz')['arr_0'], device=device)

relative_lower_bound = 0.2
output_filename_fn = lambda layer: f'{output_folder}/res_jb_sae_feature_similarity_{measure_name}_{number_of_token_desc}_{relative_lower_bound:.1f}_{layer}.npz'

d_sae = saes[0].cfg.d_sae

for layer in [9, 10]:
    print(f'Layer {layer}:')
    absolute_lower_bound = relative_lower_bound * max_activations
    aggregator = get_aggregator(measure_name)(layer, (d_sae, d_sae), lower_bound=absolute_lower_bound, device=device)

    similarities = run_with_aggregator(model, saes, 'hook_resid_pre', tokens, aggregator, device)

    np.savez_compressed(output_filename_fn(layer), similarities.cpu().numpy())
