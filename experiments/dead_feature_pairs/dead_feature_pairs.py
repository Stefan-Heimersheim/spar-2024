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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import einops
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data
from similarity_measures import DeadFeaturePairsAggregator


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
# For each pair of layers, run model with aggregator and save intermediate results
measure_name = 'pair_co_activation'
batch_size = 32
hook_name = 'hook_resid_pre'

output_folder = measure_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

n_features_1 = 100
n_features_2 = 24576
activity_lower_bound = 0.0
co_activation_thresholds = [0, 10]
evaluation_frequency = 16  # Evaluate number of empty-overlap pairs every X batches
output_filename = f'../../artefacts/{output_folder}/res_jb_sae_{measure_name}_{number_of_token_desc}_{n_features_1}_{n_features_2}_{activity_lower_bound}.npz'

d_sae = saes[0].cfg.d_sae

aggregators = [DeadFeaturePairsAggregator(layer, (n_features_1, n_features_2), lower_bound=activity_lower_bound) for layer in range(model.cfg.n_layers - 1)]

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

dead_feature_pairs = []
with torch.no_grad():
    for index, batch_tokens in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.run_with_hooks(batch_tokens)

        for aggregator in aggregators:
            # Now we can use sae_activations
            aggregator.process(sae_activations)

        # At certain points, evaluate the current overlap
        if (index + 1) % evaluation_frequency == 0:
            dead_feature_pairs.append([[(aggregator.sums <= threshold).sum() for threshold in co_activation_thresholds] for aggregator in aggregators])

    # Save overlap data
    np.savez_compressed(output_filename, np.array(dead_feature_pairs).T)


# %%
# Load and plot stats
output_folder = 'pair_co_activation'
co_activation_thresholds = [0, 5, 10, 50, 100]
evaluation_frequency = 16 * 32 * 128
n_layers = 12
n_features_1 = 100
n_features_2 = 24576

dead_feature_pairs = np.load(f'../../artefacts/{output_folder}/res_jb_sae_{output_folder}_17.5M_100_24576_0.0.npz')['arr_0']

# All features are dead at the start
dead_feature_pairs = np.concatenate([np.ones((len(co_activation_thresholds), n_layers - 1, 1)) * n_features_1 * n_features_2, dead_feature_pairs], axis=-1)

# For each threshold, add mean over all layers
dead_feature_pairs = np.concatenate([dead_feature_pairs, dead_feature_pairs.mean(axis=1, keepdims=True)], axis=1)

fig, ax = plt.subplots()
colormap = LinearSegmentedColormap.from_list('red_to_green', [(1, 0, 0, 0.7), (0, 1, 0, 0.7)], N=n_layers-1)
colors = [colormap(x) for x in np.linspace(0, 1, n_layers-1)]

threshold_index = 2  # <=10 co-activations
n_steps = dead_feature_pairs.shape[-1]

for layer in range(n_layers-1):
    ax.plot(np.arange(n_steps) * evaluation_frequency, dead_feature_pairs[threshold_index, layer, :] / (n_features_1 * n_features_2), label=f'Layers {layer}/{layer+1}', color=colors[layer])

# Plot mean over all layers
ax.plot(np.arange(n_steps) * evaluation_frequency, dead_feature_pairs[threshold_index, -1, :] / (n_features_1 * n_features_2), label=f'Mean', color='black', linewidth=2.0)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1e6)}M'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2e6))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.0f}%'))

plt.legend()
plt.xlabel('Number of tokens')
plt.ylabel(f'Number of feature pairs with <= {co_activation_thresholds[threshold_index]} co-activations')
plt.title(f'Number of SAE feature pairs that rarely co-activate')
plt.show()


# %%
# Retrieve numbers for "Number of tokens required" section in paper
token_thresholds = [1e6, 5e6, 10e6, 17.4e6]

average_dead_feature_pairs = dead_feature_pairs.mean(axis=1) / (n_features_1 * n_features_2)

average_dead_feature_pairs.shape

for index, co_activation_threshold in enumerate(co_activation_thresholds):
    for token_threshold in token_thresholds:
        number_of_tokens = int(token_threshold // evaluation_frequency)
        print(f'After {token_threshold} tokens, {(1 - average_dead_feature_pairs[index, number_of_tokens]) * 100:.1f}% of the feature pairs have more than {co_activation_threshold} co-activations.')

    print()


# %%
# Cause of dead SAE feature pairs:
# - For dead_1 dead features in layer l and dead_2 dead features in layer (l + 1), we should
#   have at least n_features_1 * n_features_2 - (n_features_1 - dead_1) * (n_features_2 - dead_2)
#   dead feature pairs between layers l and (l + 1).
# - Since the number of dead features is only available over the entire layer,
#   but we don't know how many of these are among the first 100 features,
#   we need to estimate these numbers by scaling them down

dead_features = np.load(f'../../artefacts/dead_features/res_jb_sae_dead_features_17.5M_0.0.npz')['arr_0']
dead_features = np.concatenate([np.ones((1, 9, 12)) * 24576, dead_features], axis=0)

# Only use first threshold (i.e., 0) and transpose the tensor to match dead_feature_pairs
dead_features = dead_features[:, 0, :].T

n_features_1 = 100
n_features_2 = 24576

fig, ax = plt.subplots()
colormap = LinearSegmentedColormap.from_list('red_to_green', [(1, 0, 0, 0.7), (0, 1, 0, 0.7)], N=n_layers-1)
colors = [colormap(x) for x in np.linspace(0, 1, n_layers-1)]

for layer in range(11):
    dead_1 = dead_features[layer] * n_features_1 / 24576
    dead_2 = dead_features[layer + 1]

    print(f'Layer {layer}: {dead_1[:10]=}, {dead_2[:10]=}')
    total = dead_feature_pairs[0, layer, :]
    explained_by_dead_features = (n_features_1 * n_features_2) - ((n_features_1 - dead_1) * (n_features_2 - dead_2))
    ratio = explained_by_dead_features / total

    ax.plot(np.arange(n_steps) * evaluation_frequency, ratio, label=f'Layers {layer}/{layer+1}', color=colors[layer])

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1e6)}M'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2e6))

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.0f}%'))

plt.legend()
plt.xlabel('Number of tokens')
plt.ylabel(f'% explained by individual dead features')
plt.title(f'Cause of dead SAE feature pairs')
plt.show()


# %%
# Dead SAE feature pairs that are not explained by individual dead features

dead_features = np.load(f'../../artefacts/dead_features/res_jb_sae_dead_features_17.5M_0.0.npz')['arr_0']
dead_features = np.concatenate([np.ones((1, 9, 12)) * 24576, dead_features], axis=0)

# Only use first threshold (i.e., 0) and transpose the tensor to match dead_feature_pairs
dead_features = dead_features[:, 0, :].T

n_features_1 = 100
n_features_2 = 24576

fig, ax = plt.subplots()
colormap = LinearSegmentedColormap.from_list('red_to_green', [(1, 0, 0, 0.7), (0, 1, 0, 0.7)], N=n_layers-1)
colors = [colormap(x) for x in np.linspace(0, 1, n_layers-1)]

for layer in range(11):
    dead_1 = dead_features[layer] * n_features_1 / 24576
    dead_2 = dead_features[layer + 1]

    print(f'Layer {layer}: {dead_1[:10]=}, {dead_2[:10]=}')
    total = dead_feature_pairs[0, layer, :]
    explained_by_dead_features = (n_features_1 * n_features_2) - ((n_features_1 - dead_1) * (n_features_2 - dead_2))
    difference = total - explained_by_dead_features

    ax.plot(np.arange(n_steps) * evaluation_frequency, difference, label=f'Layers {layer}/{layer+1}', color=colors[layer])

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1e6)}M'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2e6))

# ax.set_ylim(0, 1)
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.0f}%'))

plt.legend()
plt.xlabel('Number of tokens')
plt.ylabel(f'Number of DFP not explained by individual DF')
plt.title(f'Unexplained dead SAE feature pairs')
plt.show()