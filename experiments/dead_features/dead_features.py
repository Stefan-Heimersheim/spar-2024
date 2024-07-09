# Find out how many tokens are needed to get a good representation of
# SAE feature activations
#
# Idea:
# - SAE activations are sparse, so we need lots of tokens to get a 
#   sufficient number of non-zero activations for any given feature
# - This is exacerbated by the fact that we're looking at feature pairs:
#   For each pair, we need a sufficient number of tokens where _both_
#   features are active
# - In this experiment, we simply count the number of dead features for
#   a varying number of tokens
# - We define different thresholds (0, 5, 10, 50, ...) for the definition
#   of "dead" (the main plot uses threshold=0)


# %%
import os
# OPTIONAL: Set environment variable to control visibility of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import einops
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

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
class DeadFeatureAggregator:
    def __init__(self, n_layers: int, d_sae: int, lower_bound: float = 0.0):
        self.lower_bound = lower_bound

        self.counts = torch.zeros(n_layers, d_sae, dtype=torch.int)

    def process(self, sae_activations):
        active = (sae_activations > self.lower_bound)

        self.counts += active.sum(dim=-1)

    def finalize(self):
        return self.counts


# %%
# Run model with aggregator and save intermediate results
measure_name = 'dead_features'
batch_size = 32
hook_name = 'hook_resid_pre'

output_folder = measure_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

activity_lower_bound = 0.0
activity_thresholds = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000]
evaluation_frequency = 16  # Evaluate number of empty-overlap pairs every X batches
output_filename = f'{output_folder}/res_jb_sae_{measure_name}_{number_of_token_desc}_{activity_lower_bound}.npz'

n_layers = model.cfg.n_layers
d_sae = saes[0].cfg.d_sae

aggregator = DeadFeatureAggregator(n_layers, d_sae, lower_bound=activity_lower_bound)

data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

context_size = saes[0].cfg.context_size
sae_activations = torch.empty(n_layers, d_sae, batch_size * context_size)


def retrieval_hook(activations, hook):
    layer = hook.layer()

    sae_activations[layer] = einops.rearrange(
        saes[layer].encode(activations), "batch seq features -> features (batch seq)"
    )


model.add_hook(lambda name: name.endswith(f".{hook_name}"), retrieval_hook)

dead_features = []
with torch.no_grad():
    for index, batch_tokens in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.run_with_hooks(batch_tokens)

        # Now we can use sae_activations
        aggregator.process(sae_activations)

        # At certain points, evaluate the current overlap
        if (index + 1) % evaluation_frequency == 0:
            dead_features.append([(aggregator.counts <= threshold).sum(dim=-1) for threshold in activity_thresholds])

    dead_features = np.array(dead_features)  # (n_steps, n_thresholds, n_layers)

    # Save overlap data
    np.savez_compressed(output_filename, dead_features)


# %%
# Load and plot stats
output_folder = 'dead_features'
activation_thresholds = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000]
evaluation_frequency = 16 * 32 * 128
n_layers = 12
d_sae = 24576

dead_features = np.load(f'{output_folder}/res_jb_sae_dead_features_17.5M_0.0.npz')['arr_0']

# All features are dead at the start
dead_features = np.concatenate([np.ones((1, len(activation_thresholds), n_layers)) * d_sae, dead_features], axis=0)

fig, ax = plt.subplots()
colormap = LinearSegmentedColormap.from_list('red_to_green', [(1, 0, 0, 0.7), (0, 1, 0, 0.7)], N=n_layers)
colors = [colormap(x) for x in np.linspace(0, 1, n_layers)]

threshold_index = 0
n_steps = len(dead_features)

for layer in range(n_layers):
    ax.plot(np.arange(n_steps) * evaluation_frequency, dead_features[:, threshold_index, layer], label=f'Layer {layer}', color=colors[layer])

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1e6)}M'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2e6))

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1e3)}k'))

plt.legend()
plt.xlabel('Number of tokens')
plt.ylabel(f'Number of features with <= {activation_thresholds[threshold_index]} activations')
plt.title(f'Number of dead SAE features')
plt.show()
