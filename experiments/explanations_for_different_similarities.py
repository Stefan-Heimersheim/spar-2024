# For a given similarity measure, create approximately uniform samples
# of feature pairs and output them with their respective explanations.
#
# To handle the huge number (6B) of pairs, we sample at various stages of 
# the process.
#
# The output is formatted as a LaTeX table.
#
# Usage: You need to adjust lines 29-31, 34, and 36 to select the correct
# similarity measure.

# %%
# Imports
import os
import numpy as np
import sys
import pickle
import random
from collections import defaultdict
from tqdm import trange

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from similarity_helpers import load_similarity_data, get_filename


# %%
artefacts_folder = '../artefacts'
input_artefact = 'feature_similarity'
measure_name = 'sufficiency_relative_activation'  # Adjust
measure_display_name = 'sufficiency'  # Adjust
measure_ref_name = 'sufficiency'  # Adjust
sae_name = 'res_jb_sae'
n_layers = 12
d_sae = 24576

activation_threshold = 0.2  # Adjust
tokens = '10M'  # Adjust


# %%
# Load (unclamped) similarity matrix
print('Loading similarity matrix...')

input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
similarities = load_similarity_data(input_files)


# %%
# Load explanations
with open(f'../artefacts/explanations/{sae_name}_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)


# %%
# Define sampling function
def bin_and_sample_large_array_filtered(arr, bins, samples_per_bin=5, filter_ratio=0.1):
    # Calculate the number of rows to keep in 2nd and 3rd dimensions
    rows_to_keep_2d = max(1, int(arr.shape[1] * filter_ratio))
    rows_to_keep_3d = max(1, int(arr.shape[2] * filter_ratio))
    
    # Randomly select indices for 2nd and 3rd dimensions
    selected_indices_2d = np.sort(np.random.choice(arr.shape[1], rows_to_keep_2d, replace=False))
    selected_indices_3d = np.sort(np.random.choice(arr.shape[2], rows_to_keep_3d, replace=False))
    
    # Dictionary to store indices for each bin
    bin_indices = defaultdict(list)
    
    # Iterate through the filtered array
    for i in trange(arr.shape[0]):
        for j in selected_indices_2d:
            for k in selected_indices_3d:
                value = arr[i, j, k]
                bin_num = np.digitize(value, bins) - 1
                bin_indices[bin_num].append((i, j, k))
    
    # Sample from each bin
    sampled_indices = {}
    sampled_values = {}
    
    for bin_num, indices in bin_indices.items():
        if len(indices) > samples_per_bin:
            sampled = np.random.choice(len(indices), samples_per_bin, replace=False)
            sampled_indices[bin_num] = [indices[i] for i in sampled]
        else:
            sampled_indices[bin_num] = indices
        
        sampled_values[bin_num] = [arr[idx] for idx in sampled_indices[bin_num]]
    
    return sampled_indices, sampled_values


# %%
# Separate values into bins and sample from each bin
lower_bound, upper_bound, bin_width = -1, 1, 0.1
bins = np.arange(lower_bound, upper_bound + bin_width, bin_width)
sampled_indices, sampled_values = bin_and_sample_large_array_filtered(similarities, bins, samples_per_bin=2)


# %%
# Sanity check for value/index mapping
for i in range(len(bins) - 1):
    if i in sampled_indices:
        for index, value in zip(sampled_indices[i], sampled_values[i]):
            assert value == similarities[index]


# %%
# Create table with similarity values and explanations
table = f'\\begin{{table}}[htp]\n\t\\begin{{tabularx}}{{\\textwidth}}{{Y{{0.1}}Y{{0.4}}Y{{0.4}}}}\n\t\\toprule\n\t\\textbf{{Similarity}} & \\textbf{{Upstream feature}} & \\textbf{{Downstream feature}} \\\\\n\t\\midrule\n'

for i, (l, u) in enumerate(zip(bins, bins[1:])):
    # print(i, f'{l:.1f}', f'{u:.1f}')
    if i in sampled_indices:
        for index, value in zip(sampled_indices[i], sampled_values[i]):
            l, f1, f2 = index
            table += f'\t\t{value:.3f} & \\nplink{{{l}}}{{{f1}}} ({explanations[l][f1]}) & \\nplink{{{l+1}}}{{{f2}}} ({explanations[l+1][f2]}) \\\\\n'

table += f'\t\t\\bottomrule\n\t\\end{{tabularx}}\n\t\\caption{{Explanation pairs for different values of {measure_display_name}}}\n\t\\label{{tbl:{measure_ref_name}-explanations}}\n\\end{{table}}'
table = table.replace(' "', ' \\enquote{').replace('"', '}')

print(table)
