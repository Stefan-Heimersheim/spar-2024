# %%
# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import numpy as np
import sys
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from numba import jit
from collections import Counter
import networkx as nx
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data, get_filename
from visualization import show_explanation_graph


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
measure_name = 'activation_cosine_similarity'
sae_name = 'res_jb_sae'
n_tokens = '1M'
activation_threshold = 0.0
artefact_name = 'feature_similarity'

# Load similarities from unclamped files to avoid clamping errors
similarities = load_similarity_data([f'../../artefacts/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, artefact_name, activation_threshold, None, n_tokens, layer)}.npz' for layer in range(11)])
np.nan_to_num(similarities, copy=False)

# Load explanations
with open(f'../../artefacts/explanations/{sae_name}_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)


# %%
# Filter perfect (i.e., above threshold) similarities
threshold = 0.99

indices = np.where(np.abs(similarities) >= threshold)

# Convert the result to a list of 3D indices
indices_list = list(zip(*indices))


# %%
def format_neuronpedia_link(text, layer, feature):
    return f'[{text}](https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feature})'

# Save file with current date/time since there are no other identifiers
output_folder = '../../artefacts/near_perfect_similarities'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


with open(f'{output_folder}/near_perfect_similarities_{measure_name}_{threshold:.3f}.md', 'w') as f:
    f.write(f'# Explanations of near-perfect {measure_name} similarities ({len(indices_list)} pairs)\n')
    for layer, f1, f2 in tqdm(indices_list):
        f.write(f'### {measure_name}({layer}_{f1}, {layer+1}_{f2}) = {similarities[layer, f1, f2]:.4f}\n')
        f.write(f'- Feature {layer}_{f1} represents ' + format_neuronpedia_link(explanations[layer][f1], layer, f1) + '\n')
        f.write(f'- Feature {layer+1}_{f2} represents ' + format_neuronpedia_link(explanations[layer+1][f2], layer+1, f2) + '\n')    
