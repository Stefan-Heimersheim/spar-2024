# %%
from dataclasses import dataclass
import argparse
from typing import List, Dict
import torch as t
import torch
import numpy as np
import typing
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
from functools import partial
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src import D_SAE
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# %%
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"

# %%
pearson_corr_filename = f"artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz"
with open(pearson_corr_filename, 'rb') as data:
    interaction_data = np.load(data)['arr_0']

# %%    
ablation_data = np.load("/Users/benlerner/work/spar-2024/artefacts/ablations/pearson/final__toks_1048576.npz")['arr_0']
num_layers = ablation_data.shape[0]
# %%
# I need to find the indices of the ablated pairs, so that I can look up the corresponding values in the interation data
indices = np.load(f"artefacts/sampled_interaction_measures/pearson_correlation/count_75.npz")['arr_0']

# %%
# Create the new matrix
sampled_interaction_values = np.zeros(ablation_data.shape)

# Extract the values
for i in range(num_layers):
    sampled_interaction_values[i] = interaction_data[i, indices[i, :, 0], indices[i, :, 1]]


# %%
plt.clf()
similarity_scores = sampled_interaction_values
mean_diffs = ablation_data
fig, ax = plt.subplots(figsize=(12, 8))
plt.xlabel("Pearson correlation")
plt.ylabel("Mean diff in ablated score (log scale)")

cmap = cm.rainbow
color_norm = plt.Normalize(vmin=0, vmax=num_layers-1)
plt.yscale('log')
for i in range(num_layers):
    color = cmap(color_norm(i))
    plt.scatter(similarity_scores[i, :], mean_diffs[i, :], c=[color], alpha=0.7, label=f'Layer {i}')

# Create colorbar using the figure and specify the location
cbar = fig.colorbar(sm, ax=ax, ticks=range(num_layers))
cbar.set_label('Layer Index')
plt.tight_layout()
plt.show()
