# Find boolean relationships between SAE features
#
# Reasoning
# - Consider near-perfect similarities according to the necessity measure
# - Assume that we have a feature D with two upstream neighbors U1 and U2
# - This means that (approximately) whenever D is active, both U1 and U2 are active as well
# - Therefore, we have an AND gate
#
# Similarly, using the sufficiency measure finds OR gates


# %%
# Imports
import os
import numpy as np
import sys
import pickle
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import load_similarity_data, get_filename


# %%
artefacts_folder = '../../artefacts'
input_artefact = 'feature_similarity'
sae_name = 'res_jb_sae'
n_layers = 12
d_sae = 24576

activation_threshold = 0.2
tokens = '10M'


# %%
# Load (unclamped) similarity matrices
print('Loading similarity matrix (1/2)...')

measure_name = 'necessity_relative_activation'
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
necessity_similarities = load_similarity_data(input_files)
np.nan_to_num(necessity_similarities, copy=False)

print('Loading similarity matrix (2/2)...')

measure_name = 'sufficiency_relative_activation'
input_files = [f'{artefacts_folder}/similarity_measures/{measure_name}/.unclamped/{get_filename(measure_name, input_artefact, activation_threshold, None, tokens, layer)}.npz' for layer in range(n_layers - 1)]
sufficiency_similarities = load_similarity_data(input_files)
np.nan_to_num(sufficiency_similarities, copy=False)

print('Done!')


# %%
# Load explanations
with open(f'../../artefacts/explanations/{sae_name}_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)


# %%
# Look for downstream features with multiple (but few) near-perfect upstream necessity similarities
# (i.e., multiple upstream features need to be active to activate the downstream feature)
# -> Find AND gates

measure_name = 'necessity_relative_activation'
similarity_threshold = 0.999

high_similarities = (necessity_similarities >= similarity_threshold)
number_of_high_similarity_upstream_neighbors = high_similarities.sum(axis=1)

selection = (2 <= number_of_high_similarity_upstream_neighbors) & (number_of_high_similarity_upstream_neighbors <= 2)
print(f'{selection.sum()=}')

indices = np.where(selection)
indices_list = list(zip(*indices))

# Output explanations
for upstream_layer, downstream_feature in random.sample(indices_list, 100):
    upstream_features = np.where(necessity_similarities[upstream_layer, :, downstream_feature] >= similarity_threshold)[0]

    print(f'{measure_name} >= {similarity_threshold}.')
    print(f'Upstream:')
    print('\n'.join([f'{upstream_layer}_{upstream_feature} ({explanations[upstream_layer][upstream_feature]})' for upstream_feature in upstream_features]))

    print(f'Downstream:')
    print(f'{upstream_layer + 1}_{downstream_feature} ({explanations[upstream_layer + 1][downstream_feature]})')

    print()


# %%
# Look for downstream features with multiple (but few) near-perfect upstream sufficiency similarities
# (i.e., at least one of multiple upstream features needs to be active to activate the downstream feature)
# -> Find OR gates

measure_name = 'sufficiency_relative_activation'
similarity_threshold = 0.999

high_similarities = (sufficiency_similarities >= similarity_threshold)
number_of_high_similarity_upstream_neighbors = high_similarities.sum(axis=1)

selection = (2 <= number_of_high_similarity_upstream_neighbors) & (number_of_high_similarity_upstream_neighbors <= 2)
print(f'{selection.sum()=}')

indices = np.where(selection)
indices_list = list(zip(*indices))

# Output explanations
for upstream_layer, downstream_feature in random.sample(indices_list, 100):
    upstream_features = np.where(sufficiency_similarities[upstream_layer, :, downstream_feature] >= similarity_threshold)[0]

    print(f'{measure_name} >= {similarity_threshold}.')
    print(f'Upstream:')
    print('\n'.join([f'{upstream_layer}_{upstream_feature} ({explanations[upstream_layer][upstream_feature]})' for upstream_feature in upstream_features]))

    print(f'Downstream:')
    print(f'{upstream_layer + 1}_{downstream_feature} ({explanations[upstream_layer + 1][downstream_feature]})')

    print()
