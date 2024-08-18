# %%
# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import einops
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data
from similarity_helpers import clamp_low_values, save_compressed, load_similarity_data

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
# number_of_batches, number_of_token_desc = 4, '16k'
number_of_batches, number_of_token_desc = 32, '128k'
# number_of_batches, number_of_token_desc = 256, '1M'
# number_of_batches, number_of_token_desc = 4269, '17.5M'


# %%
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)
tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=number_of_batches)

# %%
# Run model with and without ablation and get difference at selected SAE features
def run_ablation(model, saes, hook_name, tokens, ablation_pos, ablation_value, measurement_pos, batch_size=32):
    data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

    context_size = saes[0].cfg.context_size
    ablation_layer, ablation_feature = ablation_pos

    def ablation_hook(activations, hook):
        sae_activations = saes[ablation_layer].encode(activations)

        if ablate:
            sae_activations[:, :, ablation_feature] = ablation_value
        
            return saes[ablation_layer].decode(sae_activations)
        
        return torch.zeros_like(activations)

    measurements = [torch.empty(len(features), batch_size * context_size) for features in measurement_pos]

    def measurement_hook(activations, hook):
        layer = hook.layer()

        sae_activations = einops.rearrange(
            saes[layer].encode(activations), "batch seq features -> features (batch seq)"
        )

        measurements[layer] = sae_activations[measurement_pos[layer], :]


    model.reset_hooks()
    model.add_hook(f"blocks.{ablation_layer}.{hook_name}", ablation_hook)
    model.add_hook(lambda name: name.endswith(f".{hook_name}"), measurement_hook)


    ablate = False
    measurements_without_ablation = []
    with torch.no_grad():
        for batch_tokens in tqdm(data_loader):
            model.run_with_hooks(batch_tokens)

            measurements_without_ablation.append(measurements)

    ablate = True
    measurements_with_ablation = []
    with torch.no_grad():
        for batch_tokens in tqdm(data_loader):
            model.run_with_hooks(batch_tokens)

            measurements_with_ablation.append(measurements)

    # Re-shape measurements
    measurements_without_ablation = [torch.concat(layer_measurements, dim=1) for layer_measurements in zip(*measurements_without_ablation)]
    measurements_with_ablation = [torch.concat(layer_measurements, dim=1) for layer_measurements in zip(*measurements_with_ablation)]

    # Calculate effect on measurement features per layer
    ablation_effect = [layer_measurements_wo - layer_measurements_w for layer_measurements_wo, layer_measurements_w in zip(measurements_without_ablation, measurements_with_ablation)]
    
    return ablation_effect

# %%
ablation_pos = 6, 1234
measurement_pos = [[], list(range(1000)), [], [], [], [], [], [], [], [], [], []]

ablation_effect = run_ablation(model, saes, 'hook_resid_pre', tokens, ablation_pos, 0, measurement_pos)
[(effect.shape, effect.max()) for effect in ablation_effect if len(effect) > 0]

# %%
# For a given upstream feaure, find the most similar downstream neighbors
# and compare ablation effects with random downstream neighbors

pearson_correlation = load_similarity_data(['../../artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz'])


# %%
def get_ablation_effect_similar_vs_random(model, saes, hook_name, tokens, ablation_pos, similarities, number_of_measured_features):
    n_layers = model.cfg.n_layers
    d_sae = saes[0].cfg.d_sae
    
    
    ablation_layer, ablation_feature = ablation_pos
    measurement_pos = [[] for _ in range(n_layers)]

    # First, measure ablation effect at most similar downstream neighbors
    most_similar_features = np.argpartition(similarities[ablation_layer, ablation_feature], -number_of_measured_features)[-number_of_measured_features:]
    measurement_pos[ablation_layer + 1] = most_similar_features

    similar_ablation_effect = run_ablation(model, saes, hook_name, tokens, ablation_pos, 0, measurement_pos)

    # Second, choose random features
    measurement_pos[ablation_layer + 1] = list(np.random.choice(d_sae, number_of_measured_features, replace=False))

    random_ablation_effect = run_ablation(model, saes, hook_name, tokens, ablation_pos, 0, measurement_pos)

    return similar_ablation_effect[ablation_layer + 1], random_ablation_effect[ablation_layer + 1]

    mean_similar_ablation_effect = similar_ablation_effect[ablation_layer + 1].mean()
    mean_random_ablation_effect = random_ablation_effect[ablation_layer + 1].mean()

    return mean_similar_ablation_effect, mean_random_ablation_effect


# %%
s, r = get_ablation_effect_similar_vs_random(model, saes, 'hook_resid_pre', tokens, (6, 0), pearson_correlation, 1000)


# %%
s.max(), r.max()