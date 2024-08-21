# %%
import os
import torch
import einops
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import pickle
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data
from similarity_helpers import load_similarity_data, get_filename
from visualization import show_explanation_graph


# %%
device = 'cuda:1'
n_samples = 100


# %%
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)
tokens = model.to_tokens('When Mary and John went to the store, John gave a drink to')

# Load similarities for edge width
measure_name = 'pearson_correlation'
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
def get_active_features(model, saes, tokens, lower_bound, relative_bound, hook_name='hook_resid_pre'):
    n_layers = model.cfg.n_layers
    n_features = saes[0].cfg.d_sae
    batch_size, context_size = tokens.shape

    if relative_bound:
        lower_bound = lower_bound.unsqueeze(2).unsqueeze(3)

    sae_activations = torch.empty(n_layers, batch_size, context_size, n_features)

    def retrieval_hook(activations, hook):
        layer = hook.layer()

        sae_activations[layer] = saes[layer].encode(activations)

    model.add_hook(lambda name: name.endswith(f".{hook_name}"), retrieval_hook)

    with torch.no_grad():
        model.run_with_hooks(tokens)

        # Now we can use sae_activations
        return einops.rearrange((einops.rearrange(sae_activations, 'n_layers n_samples n_tokens n_features -> n_layers n_features n_samples n_tokens') > lower_bound)[:, :, :, -1].bool(), 'n_layers n_features n_samples -> n_samples n_layers n_features')


# %%
# Plot number of active features for different lower bounds
max_activations = torch.tensor(np.load('../../artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz')['arr_0'])

lower_bounds = np.linspace(0, 1, 20)
mean_active_features = [get_active_features(model, saes, tokens, lower_bound=lower_bound * max_activations, relative_bound=True).sum(dim=(1, 2)).float().mean() for lower_bound in tqdm(lower_bounds)]

plt.plot(lower_bounds, mean_active_features)
plt.title(f'Mean number of active features for a single token')
plt.xlabel('Relative activation bound')
plt.ylabel('Number of active features (across all layers)')
plt.show()


# %%
# Save matrix for a specific (relative) lower bound
lower_bound = 0.5

any_active = get_active_features(model, saes, tokens, lower_bound=lower_bound * max_activations, relative_bound=True)
print(f'Mean number of active features per sample: {any_active.sum(dim=(1, 2)).float().mean()}')


# %%
# Convert matrix into graph
n_layers = 12
nodes = [any_active[0, layer].nonzero().flatten().tolist() for layer in range(n_layers)]

def add_explanations(graph):
    for node, attr in graph.nodes(data=True):
        graph.nodes[node]['explanation'] = explanations[attr['layer']][attr['feature']]


# %%
graph = nx.DiGraph()
for layer, features in enumerate(nodes):
    graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer, 'feature': feature}) for feature in features])

for layer, (features_from, features_to) in enumerate(zip(nodes, nodes[1:])):
    graph.add_edges_from([(f'{layer}_{out_feature}', f'{layer+1}_{in_feature}', {'similarity': abs(similarities[layer, out_feature, in_feature])}) for out_feature in features_from for in_feature in features_to])

add_explanations(graph)
fig = show_explanation_graph(graph, show=False)
fig.update_layout(title=f'Active features on the final token of an IOI prompt (relative activation threshold: {lower_bound})', font=dict(size=9))


# %%
folder = '../../artefacts/active_features/ioi'
Path(folder).mkdir(parents=True, exist_ok=True)

fig.write_html(f'{folder}/{sae_name}_active_features_rel_{lower_bound}_ioi.html')