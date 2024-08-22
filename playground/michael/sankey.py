# %%
# Imports
import os
import numpy as np
import sys
from tqdm import tqdm, trange
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data, get_filename


# %%
artefacts_folder = '../../artefacts'
input_artefact = 'feature_similarity'
sae_name = 'res_jb_sae'
n_layers = 12

measure_name = 'pearson_correlation'
activation_threshold = 0.0
tokens = '1M'


# %%
print('Loading similarity matrix...')
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
similarities = load_similarity_data(input_files)
np.nan_to_num(similarities, copy=False)


# %%
bound = 0.5

forward_max = similarities.max(axis=2)
forward_pass_through = (forward_max >= bound).sum(axis=1)

backward_max = similarities.max(axis=1)
backward_pass_through = (backward_max >= bound).sum(axis=1)

d_sae = 24576


# %%
forward_dying = d_sae - forward_pass_through
backward_new = d_sae - backward_pass_through

# %%
n_layers = 12
labels = [label for layer in range(n_layers) for label in ['new', f'L{layer}', 'dying']][1:-1]
pos_x = [x for layer in range(n_layers) for x in [(2 * layer - 1) / (2 * n_layers - 1), 2 * layer / (2 * n_layers - 1), (2 * layer + 1) / (2 * n_layers - 1)]][1:-1]
pos_y = [y for _ in range(n_layers) for y in [0.2, 0.5, 0.8]][1:-1]
colors = [label for layer in range(n_layers) for label in ['green', f'blue', 'red']][1:-1]

sources = [s for layer in range(n_layers - 1) for s in [3 * layer, 3 * layer, 3 * layer + 2]]
targets = [t for layer in range(n_layers - 1) for t in [3 * layer + 1, 3 * layer + 3, 3 * layer + 3]]
values = [v for layer in range(n_layers - 1) for v in [forward_dying[layer], forward_pass_through[layer], backward_new[layer]]]


# %%
fig = go.Figure(go.Sankey(
    arrangement = "fixed",
    node = {
        "label": labels,
        "x": pos_x,
        "y": pos_y,
        'pad':10,  # 10 Pixels,
        'color': colors
    },
    link = {
        "source": sources,
        "target": targets,
        "value": values
    }))

fig.update_layout(
    autosize=False,
    width=100 * n_layers,  # Specify width in pixels
    height=500   # Specify height in pixels
)

fig.show()


# %%
