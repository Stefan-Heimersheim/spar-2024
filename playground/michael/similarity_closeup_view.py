# %%
# Imports
import os
import numpy as np
import sys
from tqdm import tqdm, trange
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data, get_filename


# %%
def plot_feature_connections(measure_name, histogram_thresholds, input_files, output_file):
    print('Loading similarity matrix...')
    similarities = load_similarity_data(input_files)

    fig = go.Figure()

    layers = [f'{layer} -> {layer+1}' for layer in range(n_layers - 1)]
    for threshold in histogram_thresholds:
        print('.', end='')
        values = (similarities >= threshold).sum(axis=(1, 2))
        fig.add_trace(go.Bar(x=layers, y=values, name=f'm >= {threshold}'))

    print()

    # Update layout for logarithmic y-axis
    fig.update_layout(
        title=f'Feature connection histogram (m = {measure_name})',
        xaxis_title='Layer pair',
        yaxis_title='Number of feature pairs',
        yaxis_type='log'
    )

    # Show the plot
    fig.show()
    fig.write_html(output_file)

                   
# %%
sae_name = 'res_jb_sae'
n_layers = 12
measure_names = ['activation_cosine_similarity', 'cosine_similarity', 'jaccard_similarity', 'mutual_information', 'necessity', 'pearson_correlation', 'sufficiency']
activation_thresholds = [0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0]
tokens = ['1M', None, '1M', '1M', '1M', '1M', '1M']
histogram_thresholds = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1]
artefacts_folder = '../../artefacts'
input_artefact = 'feature_similarity'
output_artefact = 'feature_similarity_histogram'

output_folder = '../../artefacts/feature_similarity_histograms'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for measure_name, activation_threshold, n_tokens in tqdm(list(zip(measure_names, activation_thresholds, tokens))):
    input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, n_tokens, layer)}.npz' for layer in range(n_layers - 1)]
    output_file = f'{artefacts_folder}/feature_similarity_histograms/{get_filename(measure_name, output_artefact, activation_threshold, None, n_tokens)}.html'
    
    plot_feature_connections(measure_name, histogram_thresholds, input_files, output_file)
