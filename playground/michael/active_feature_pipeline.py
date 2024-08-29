# %%
import os
import sys
import torch
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data
from similarity_helpers import load_similarity_data, get_filename
from visualization import get_active_feature_graph_for_prompt, save_graph_to_json


# %%
# Init device and folder
if torch.cuda.is_available():
    device = 'cuda:3'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

artefacts_folder = '../../artefacts'


# %%
# Init model, SAEs and similarity matrix
sae_name = 'res_jb_sae'
measure_name = 'pearson_correlation'
measure_display_name = 'pearson correlation'
activation_threshold_1 = None
clamping_threshold = 0.1
activation_threshold_2 = 0.5
n_tokens = '10M'

model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)
similarities = load_similarity_data([f'{artefacts_folder}/similarity_measures/{measure_name}/{get_filename(measure_name, "feature_similarity", activation_threshold_1, clamping_threshold, n_tokens)}.npz'])


# %%
# Create graph for specific prompt
prompt = """
When trying to understand the inner workings of Large Language models (LLMs), it is striking that the contents of a transformer’s residual stream are difficult to interpret. However, Sparse Autoencoders (SAEs) have shown promising progress by projecting the residual stream into a higher-dimensional space where interpretable concepts (features) are represented as single directionsdimensions. In this paper, we aim to provide an understanding of how the features of SAEs in different layers of the model are related by building a graph relating features between adjacent layers. We show that these measures might be suited for automated interpretability research, and provide examples of macroscopic structural patterns across many features, as well as individual feature relationships that can be uncovered in GPT-2-small by this novel approach.
"""

graph = get_active_feature_graph_for_prompt(model, saes, prompt, similarities, activation_threshold_2, artefacts_folder=artefacts_folder, verbose=True)
graph.graph['description'] = f'This graph\'s nodes are the SAE features that are active (i.e., whose activation is {activation_threshold_2 or 0} or higher) on the final token of the prompt. Its edges represent the similarity values of the {measure_name} measure, computed over {n_tokens} tokens with activation threshold {activation_threshold_1} (absolute values below {clamping_threshold} are clamped to zero). The explanations of the features are created by GPT-3.5-turbo and downloaded from Neuronpedia.'
graph.graph['prompt'] = prompt

# Save graph to file
save_graph_to_json(graph, f'{artefacts_folder}/active_feature_graphs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{sae_name}_active_feature_graph_{measure_name}_.json')
