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
import pandas as pd

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
MEASURE = 'jaccard'
MEASURE_READABLE = 'Jaccard'
# %%    
# ablation_0_1 = np.load("artefacts/ablations/pearson/count_90__up_til_layer_1__toks_1048576.npz")['arr_0']
# ablation_2_6 = np.load("artefacts/ablations/pearson/count_90__up_til_layer_6__toks_1048576.npz")['arr_0']
# ablation_7_10 = np.load("artefacts/ablations/pearson/count_90__final__toks_1048576.npz")['arr_0']
# ablation_data = np.concatenate([ablation_0_1[:2], ablation_2_6[2:7], ablation_7_10[7:]], axis=0)
ablation_data = np.load(f"artefacts/ablations/jaccard/count_100__final__toks_1048576.npz")['arr_0']
# %%
num_layers = ablation_data.shape[0]
# %%
# I need to find the indices of the ablated pairs, so that I can look up the corresponding values in the interation data
indices = np.load(f"artefacts/sampled_interaction_measures/jaccard_similarity/count_100.npz")['arr_0']

# %%
def matrix_summary_statistics(matrix):
    # Convert the matrix to a 1D array
    flat_matrix = np.array(matrix).flatten()
    
    # Create a pandas Series from the flattened matrix
    series = pd.Series(flat_matrix)
    
    # Calculate summary statistics
    summary = series.describe(percentiles=[0.2, 0.4, 0.6, 0.8])
    
    return summary

# %%
def create_boxplots(xs, ys, bins):
    fig, ax = plt.subplots(figsize=(15, 8))
    
    boxplot_data = []
    positions = []
    mean_values = []
    
    for i in range(len(bins) - 1):
        lower_bound, upper_bound = bins[i], bins[i+1]
        mask = (xs >= lower_bound) & (xs < upper_bound)
        bin_data = ys[mask]
        
        if bin_data.size > 0:
            boxplot_data.append(bin_data)
            positions.append(i)
            mean_values.append(np.mean(bin_data))
    
    bp = ax.boxplot(boxplot_data, positions=positions, vert=True, 
                    patch_artist=True, whis=[0, 100])
    
    # Customize the appearance
    for box in bp['boxes']:
        box.set(facecolor='lightblue', edgecolor='blue', alpha=0.7)
    for whisker in bp['whiskers']:
        whisker.set(color='blue', linewidth=1.5, linestyle='--')
    for cap in bp['caps']:
        cap.set(color='blue', linewidth=2)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    
    # Add green lines for means
    mean_lines = []
    for i, pos in enumerate(positions):
        line = ax.hlines(mean_values[i], pos - 0.4, pos + 0.4, colors='green', linewidth=2)
        mean_lines.append(line)
    
    # Set labels and title
    ax.set_xlabel(f'{MEASURE_READABLE} bin midpoint')
    ax.set_ylabel('mean ablated diff')
    ax.set_title(f'Box Plots of Mean Ablated Diff by {MEASURE_READABLE} Bins')
    
    # Set x-ticks to be the middle of each bin
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{bin_centers[i]:.2f}' for i in positions])
    
    # Add legend
    ax.legend([bp["medians"][0], mean_lines[0]], ['Median', 'Mean'],
              loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
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
plt.xlabel(MEASURE_READABLE)
plt.ylabel("Mean diff in ablated score")

cmap = cm.rainbow
color_norm = plt.Normalize(vmin=0, vmax=num_layers-1)
for i in range(num_layers):
    color = cmap(color_norm(i))
    plt.scatter(similarity_scores[i, :], mean_diffs[i, :], c=[color], alpha=0.7, label=f'Layer {i}')

# Create colorbar using the figure and specify the location
sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, ticks=range(num_layers))
cbar.set_label('Layer Index')
plt.tight_layout()
plt.show()

# %%
create_boxplots(similarity_scores.flatten(), mean_diffs.flatten(), np.linspace(0.01, 1.0, 10))

# %%
