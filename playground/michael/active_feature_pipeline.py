# %%
import os
import sys

import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data
from similarity_helpers import load_similarity_data, get_filename
from visualization import get_active_feature_graph_for_prompt, save_graph_to_json


# %%
# Init device and folder
device = 'cuda:3'
artefacts_folder = '../../artefacts'


# %%
# Init model, SAEs and similarity matrix
sae_name = 'res_jb_sae'
measure_name = 'pearson_correlation'
activation_threshold_1 = None
clamping_threshold = 0.1
activation_threshold_2 = 0.5
n_tokens = '10M'

model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)
similarities = load_similarity_data([f'{artefacts_folder}/similarity_measures/{measure_name}/{get_filename(measure_name, "feature_similarity", activation_threshold_1, clamping_threshold, n_tokens)}.npz'])


# %%
# Create graph for specific prompt
prompt = 'Hello, how are you?'

graph = get_active_feature_graph_for_prompt(model, saes, prompt, similarities, activation_threshold_2, artefacts_folder=artefacts_folder, verbose=True)
graph.graph['description'] = f'This graph\'s nodes are the SAE features that are active (i.e., whose activation is {activation_threshold_2} or higher) on the final token of the prompt "{prompt}". Its edges represent the similarity values of the {measure_name} measure, computed over {n_tokens} tokens with activation threshold {activation_threshold_1} (absolute values below {clamping_threshold} are clamped to zero). The explanations of the features are created by GPT-3.5-turbo and downloaded from Neuronpedia.'

# Save graph to file
save_graph_to_json(graph, f'{artefacts_folder}/active_feature_graphs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{sae_name}_active_feature_graph_{measure_name}_.json')
