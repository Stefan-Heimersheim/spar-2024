# %%
# Imports
import os
import numpy as np
import sys
from tqdm import tqdm, trange
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

def get_pass_through_scores(measure_name, sae_name, activation_threshold, tokens, return_absolute_score=True, return_relative_score=True):
    # Always start with unclamped similarity matrices
    print('Loading similarity matrix...')
    input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
    similarities = load_similarity_data(input_files)

    absolute_score = None
    if return_absolute_score:
        # Return the maximum similarity value _from the previous layer to the current layer_
        absolute_score = similarities.max(axis=1)
    
    relative_score = None
    if return_relative_score:
        # Return the maximum similarity value _from the previous layer to the current layer_, divided by the sum of similarity values
        relative_score = similarities.max(axis=1) / similarities.sum(axis=1)

    return absolute_score, relative_score


# %%
# Always start with unclamped similarity matrices
print('Loading similarity matrix...')
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
similarities = load_similarity_data(input_files)


# %%
np.nan_to_num(similarities, copy=False)


# %%
absolute_threshold = 0.9
relative_threshold = 0.3

# Get the maximum similarity value _from the previous layer to the current layer_
absolute_score = similarities.max(axis=1)

# Get the maximum similarity value _from the previous layer to the current layer_, divided by the sum of similarity values
relative_score = similarities.max(axis=1) / similarities.sum(axis=1)

# %%
pass_through_features_abs = absolute_score >= absolute_threshold
pass_through_features_rel = relative_score >= relative_threshold


# %%
abs_rel = pass_through_features_abs & pass_through_features_rel
abs_no_rel = pass_through_features_abs & ~pass_through_features_rel
no_abs_rel = ~pass_through_features_abs & pass_through_features_rel
no_abs_no_rel = ~pass_through_features_abs & ~pass_through_features_rel

abs_rel_sum = abs_rel.sum(axis=1)
abs_no_rel_sum = abs_no_rel.sum(axis=1)
no_abs_rel_sum = no_abs_rel.sum(axis=1)
no_abs_no_rel_sum = no_abs_no_rel.sum(axis=1)

data_numpy = np.stack([abs_rel_sum, abs_no_rel_sum, no_abs_rel_sum, no_abs_no_rel_sum]).T


# %%
fig = go.Figure()

categories = [f'{layer}->{layer+1}' for layer in range(n_layers - 1)]
stack_labels = ['yes/yes', 'yes/no', 'no/yes', 'no/no']

for i in range(data_numpy.shape[1]):
    fig.add_trace(go.Bar(
        x=categories,
        y=data_numpy[:, i],
        name=stack_labels[i]
    ))

fig.update_layout(
    title='Pass-through features (absolute/relative)',
    xaxis_title='Layer',
    yaxis_title='Values',
    barmode='stack'
)

fig.show()

# %%
def get_diff(rel_value):
    relative_score = similarities.max(axis=1) / similarities.sum(axis=1)
    pass_through_features_rel = relative_score >= rel_value

    abs_no_rel = pass_through_features_abs & ~pass_through_features_rel
    no_abs_rel = ~pass_through_features_abs & pass_through_features_rel

    abs_no_rel_sum = abs_no_rel.sum(axis=1)
    no_abs_rel_sum = no_abs_rel.sum(axis=1)

    return (abs_no_rel_sum + no_abs_rel_sum).mean()

rel_values = np.linspace(0, 1, 30)
diffs = [get_diff(rel_value) for rel_value in tqdm(rel_values)]

plt.plot(rel_values, diffs)


# %%
    