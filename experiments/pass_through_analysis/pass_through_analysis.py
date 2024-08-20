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
def similarity_pass_through(similarities):
    pass_through_bounds = np.linspace(0, 1, 100)

    # Forward pass-through
    forward_max = similarities.max(axis=2)

    forward_fig = go.Figure()
    for layer in range(n_layers - 1):
        forward_fig.add_trace(go.Scatter(
            x=pass_through_bounds,
            y=[(forward_max[layer] >= bound).sum() for bound in pass_through_bounds],
            mode='lines',
            name=f'Layer {layer}->{layer+1}'
        ))

    forward_fig.update_layout(
        title='Number of forward pass-through features',
        xaxis_title='Pass-through bound (abs. max. similarity)',
        yaxis_title='Number of features',
    )


    # Backward pass-through
    backward_max = similarities.max(axis=1)

    backward_fig = go.Figure()
    for layer in range(n_layers - 1):
        backward_fig.add_trace(go.Scatter(
            x=pass_through_bounds,
            y=[(backward_max[layer] >= bound).sum() for bound in pass_through_bounds],
            mode='lines',
            name=f'Layer {layer+1}->{layer}'
        ))

    backward_fig.update_layout(
        title=f'Number of backward pass-through features',
        xaxis_title='Pass-through bound (abs. max. similarity)',
        yaxis_title='Number of features',
    )

    return forward_fig, backward_fig


# %%
# Pearson correlation
artefacts_folder = '../../artefacts'
input_artefact = 'feature_similarity'
sae_name = 'res_jb_sae'
n_layers = 12

measure_name = 'pearson_correlation'
activation_threshold = 0.0
tokens = '1M'

print('Loading similarity matrix...')
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
similarities = load_similarity_data(input_files)
np.nan_to_num(similarities, copy=False)

print('Plotting...')
forward_fig, backward_fig = similarity_pass_through(similarities)

forward_fig.update_layout(title=f'Number of forward pass-through features ({measure_name})')
backward_fig.update_layout(title=f'Number of backward pass-through features ({measure_name})')

forward_fig.show()
backward_fig.show()


# %%
# Cosine similarity (on decoder weights)
artefacts_folder = '../../artefacts'
input_artefact = 'feature_similarity'
sae_name = 'res_jb_sae'
n_layers = 12

measure_name = 'cosine_similarity'
activation_threshold = None
tokens = None

print('Loading similarity matrix...')
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
similarities = load_similarity_data(input_files)
np.nan_to_num(similarities, copy=False)

print('Plotting...')
forward_fig, backward_fig = similarity_pass_through(similarities)

forward_fig.update_layout(title=f'Number of forward pass-through features ({measure_name})')
backward_fig.update_layout(title=f'Number of backward pass-through features ({measure_name})')

forward_fig.show()
backward_fig.show()


# %%
# Cosine similarities on gpt-3.5-turbo explanations
print('Loading explanations...')
with open('../../artefacts/explanations/res_jb_sae_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)

print('Loading embedder...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print('Computing embeddings...')
embeddings = np.array([embedder.encode(layer_explanations) for layer_explanations in tqdm(explanations)])

similarities = np.array([cosine_similarity(embeddings[layer], embeddings[layer + 1]) for layer in trange(n_layers - 1)])

print('Computing similarities...')
forward_fig, backward_fig = similarity_pass_through(similarities)

print('Plotting...')
forward_fig.update_layout(title=f'Number of forward pass-through features (explanation similarity)')
backward_fig.update_layout(title=f'Number of backward pass-through features (explanation similarity)')

forward_fig.show()
backward_fig.show()