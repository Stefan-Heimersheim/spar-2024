# %%
import torch as t
import pandas as pd
import numpy as np
from datasets import load_dataset  
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import einops
from tqdm import tqdm, trange
import plotly.express as px
import networkx as nx
import requests


# %%
# Config
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
# Load Model and SAEs
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

blocks = list(range(len(model.blocks)))
saes = []
for block in tqdm(blocks):
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=f"blocks.{block}.hook_resid_pre", device=device)
    saes.append(sae)



# %%
# This uses way too much memory, leaving it here for reference
sae_0 = saes[0]
sae_1 = saes[1]

# Measure pairwise cos similarity between all features in consecutive layers
print(f'{sae_0.W_dec.shape=}')
W_dec_0_expanded = einops.repeat(sae_0.W_dec, 'd_sae d_model -> d_sae d_model d_model_2', d_model_2=sae_1.W_dec.shape[1])
W_dec_1_expanded = einops.repeat(sae_1.W_dec, 'd_sae d_model -> d_sae d_model d_model_2', d_model_2=sae_0.W_dec.shape[1])

cos_similarity = einops.einsum(W_dec_0_expanded, W_dec_1_expanded,
                               'd_sae_1 d_model_1 d_model_2, d_sae_2 d_model_1 d_model_2 -> d_sae_1 d_sae_2')
# %%
# Perform similarity analysis for subsequent layers

# Define correlation function
def cosine_similarity(tensor1, tensor2):
    return t.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)
# %%

# For simplicity, only consider first X features in each layer
n_features = sae.cfg.d_sae
# n_features = 200
d_model = sae.cfg.d_in

# Init similarity graph
# %%
n_layers = len(blocks)
layers = list(enumerate(blocks))

# For each pair of subsequent layers, look at each pair of features from different
# layers, calculate their similarity and add an edge in the similarity graph if it
# exceeds the threshold value
similarity_threshold = 0.3
results = t.empty([n_layers, n_features, n_features], device='cpu')
with t.no_grad():
    for (layer_1_index, layer_1_name), (layer_2_index, layer_2_name) in zip(layers, layers[1:]):
        print(f'Computing feature similarities between layers {layer_1_name} and {layer_2_name}...')
        layer_results = t.empty([n_features, n_features], device=device)
        layer_1_W_dec, layer_2_W_dec = saes[layer_1_index].W_dec, saes[layer_2_index].W_dec

        for feature_1_index, feature_1 in tqdm(enumerate(layer_1_W_dec[:n_features]), total=n_features):
            cos_similarities = t.nn.functional.cosine_similarity(feature_1.unsqueeze(0), layer_2_W_dec[:n_features], dim=1)
            layer_results[feature_1_index, :] = cos_similarities
        results[layer_1_index, :, :] = layer_results.cpu()


# %%
t.save(results, 'dani/dani-0/cosine_similarities-gpt2-small-res-jb.pt')
# %%
results = t.load('dani/dani-0/cosine_similarities-gpt2-small-res-jb.pt')
# %%
similarity_threshold = 0.5
is_similar = results > similarity_threshold
# %%
# %%
graph = nx.Graph()
graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer}) for _, layer in layers for feature in range(n_features)])

# For each pair of subsequent layers, look at each pair of features from different
# layers, calculate their similarity and add an edge in the similarity graph if it
# exceeds the threshold value
for (layer_1_index, layer_1_name), (layer_2_index, layer_2_name) in zip(layers[:1], layers[1:2]):
    for feature_1 in tqdm(range(n_features)):
        for feature_2 in range(n_features):
            if is_similar[layer_1_index, feature_1, feature_2]:
                graph.add_edge(f'{layer_1_name}_{feature_1}', f'{layer_2_name}_{feature_2}')
# %%
nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key='layer'), node_size=10)
# %%
import plotly_express as px
flat_results = results[0].flatten().numpy()
# randomly sample 1000000 elements
flat_results_sample = np.random.choice(flat_results, 1000000)
px.histogram(flat_results_sample)
# %%
