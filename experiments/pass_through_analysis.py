# %%
# Imports
import os
import numpy as np
import sys
import plotly.graph_objects as go
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from similarity_helpers import load_similarity_data, get_filename


# %%
artefacts_folder = '../artefacts'
input_artefact = 'feature_similarity'
sae_name = 'res_jb_sae'
n_layers = 12
d_sae = 24576

measure_name = 'pearson_correlation'
activation_threshold = None
tokens = '10M'


# %%
# Load (unclamped) similarity matrix
print('Loading similarity matrix...')
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
similarities = load_similarity_data(input_files)
np.nan_to_num(similarities, copy=False)


# %%
pass_through_threshold = 0.95

forward_max = similarities.max(axis=2)
forward_pass_through = (forward_max >= pass_through_threshold).sum(axis=1)

backward_max = similarities.max(axis=1)
backward_pass_through = (backward_max >= pass_through_threshold).sum(axis=1)

forward_disappearing = d_sae - forward_pass_through
backward_appearing = d_sae - backward_pass_through

relative_forward_disappearing = forward_disappearing / d_sae
relative_backward_appearing = backward_appearing / d_sae

labels = [label for layer in range(n_layers) for label in [f'appearing<br>({np.take(relative_backward_appearing, layer, mode="clip"):.1%})', f'Layer {layer}', f'disappearing<br>({np.take(relative_forward_disappearing, layer, mode="clip"):.1%})']][1:-1]
pos_x = [x for layer in range(n_layers) for x in [(2 * layer) / (2 * n_layers), (2 * layer + 1) / (2 * n_layers), (2 * layer + 2) / (2 * n_layers)]][1:-1]
pos_y = [y for _ in range(n_layers) for y in [0.2, 0.5, 0.8]][1:-1]
colors = [label for layer in range(n_layers) for label in ['green', f'blue', 'red']][1:-1]

sources = [s for layer in range(n_layers - 1) for s in [3 * layer, 3 * layer, 3 * layer + 2]]
targets = [t for layer in range(n_layers - 1) for t in [3 * layer + 1, 3 * layer + 3, 3 * layer + 3]]
values = [v for layer in range(n_layers - 1) for v in [forward_disappearing[layer], forward_pass_through[layer], backward_appearing[layer]]]


# %%
# Plot diagram
fig = go.Figure(go.Sankey(
    arrangement = "fixed",
    node = dict(label=labels, x=pos_x, y=pos_y, pad=10, color=colors),
    link = dict(source=sources, target=targets, value=values)
))

fig.update_layout(autosize=False, width=150 * n_layers, height=750)

fig.show()


# %%
# Save diagram as PNG
output_folder = '../artefacts/pass_through_analysis'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

fig.write_image(f"{output_folder}/{sae_name}_pass_through_features_{measure_name}_{tokens}_{pass_through_threshold:.1f}.png")


# %%
# Find optimal pass-through threshold by performing a binary search, 
# comparing explanations for any given threshold
measure_name = 'pearson_correlation'
activation_threshold = None
tokens = '10M'

# Load (unclamped) similarity matrix
print('Loading similarity matrix...')
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
similarities = load_similarity_data(input_files)
np.nan_to_num(similarities, copy=False)

print('Loading explanations...')
with open(f'../artefacts/explanations/{sae_name}_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)


# %%
number_of_samples = int(input('How many samples would you like to see at each step?'))

print('Starting binary search...')
lower, upper, threshold = 0.5, 1, 0.75
while True:
    print(f'Current threshold: {threshold}')

    # Create, sample, and output examples
    pass_through_features = np.array(np.where(similarities >= threshold)).T
    print(f'{len(pass_through_features)} feature pairs are above the threshold.\n')
    samples = pass_through_features[np.random.choice(len(pass_through_features), number_of_samples, replace=False)]

    print('\n\n'.join([f'{l}_{f1}: {explanations[l][f1]}\n{l+1}_{f2}: {explanations[l+1][f2]}' for l, f1, f2 in samples]))
    
    answer = input('Do all of the above examples qualify as pass-through features?')

    if answer == 'yes':
        upper = threshold
    elif answer == 'no':
        lower = threshold
    elif answer == '':
        break

    threshold = (upper + lower) / 2

    print('\n')

print(f'Narrowed threshold down to [{lower}, {upper}].')

# %%
