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
# %%

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
# %% save compressed in numpy format
np.savez_compressed('dani/dani-0/cosine_similarities-gpt2-small-res-jb.npz', results.cpu().numpy())
# %%
results = np.load('dani/dani-0/cosine_similarities-gpt2-small-res-jb.npz')['arr_0']
# %%
import michael.compress_matrix as compress_matrix
# %%
compress_matrix.clamp_low_values(results, threshold=0.2)
# %%
compress_matrix.save_compressed(results, 'dani/dani-0/cosine_similarities-gpt2-small-res-jb-clamped-0_2.npz')
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
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)
# %%
batch_size = 64

# Get model activations and compute feature activations
for sae in saes:
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with t.no_grad():
    batch_tokens = token_dataset[:batch_size]["tokens"]
    print('Running model with cache...')
    _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

    # Use the SAE
    print('Computing feature activations...')
    feature_acts = []
    for sae in tqdm(saes):
        layer_feature_acts = sae.encode(cache[sae.cfg.hook_name])
        # sae_out = sae.decode(feature_acts)
        feature_acts.append(layer_feature_acts)

    # Save some memory
    del cache
# %%
def pearson_correlation(tensor1, tensor2):
    # Ensure the tensors have the same length
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"

    # Calculate the mean of each tensor
    mean1 = t.mean(tensor1)
    mean2 = t.mean(tensor2)

    # Calculate the standard deviation of each tensor
    std1 = t.std(tensor1, unbiased=False)
    std2 = t.std(tensor2, unbiased=False)

    # Calculate the covariance between the two tensors
    covariance = t.mean((tensor1 - mean1) * (tensor2 - mean2))

    # Calculate the Pearson correlation coefficient
    correlation = covariance / (std1 * std2)

    return correlation.item()


# Naive co-occurrence function (how often do binary activations coincide?)
def co_occurrence_1(tensor1, tensor2):
    # Ensure the tensors have the same length
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"

    active_1 = tensor1 > 0
    active_2 = tensor2 > 0

    co = (active_1 == active_2).float().mean()

    return co.item()


# Co-occurrence function (how often do binary activations coincide when they're not both zero?)
def co_occurrence_2(tensor1, tensor2):
    # Ensure the tensors have the same length
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"

    active_1 = tensor1 > 0
    active_2 = tensor2 > 0

    both_non_zero = (active_1 & active_2).sum()
    one_non_zero = (active_1 | active_2).sum()

    return (both_non_zero / one_non_zero).item()

# %%128
n_layers = 1
n_features = sae.cfg.d_sae // 20
acts_per_feature = [einops.rearrange(layer_feature_acts, 'b p f -> f (b p)') for layer_feature_acts in feature_acts]
correlation_results = t.empty([n_layers, n_features, n_features], device='cpu')
co_occurrence_results = t.empty([n_layers, n_features, n_features], device='cpu')
with t.no_grad():
    for (layer_1_index, layer_1_name), (layer_2_index, layer_2_name) in zip(layers[:1], layers[1:2]):
        print(f'Computing feature similarities between layers {layer_1_name} and {layer_2_name}...')
        layer_1_acts_per_feature, layer_2_acts_per_feature = acts_per_feature[layer_1_index], acts_per_feature[layer_2_index]
        layer_correlations = t.empty([n_features, n_features], device=device)
        layer_co_occurrences = t.empty([n_features, n_features], device=device)

        for feature_1_index, feature_1 in tqdm(enumerate(layer_1_acts_per_feature[:n_features]), total=n_features):
            for feature_2_index, feature_2 in tqdm(enumerate(layer_2_acts_per_feature[:n_features]), total=n_features):
                layer_correlations[feature_1_index, feature_2_index] = pearson_correlation(feature_1, feature_2)
                layer_co_occurrences[feature_1_index, feature_2_index] = co_occurrence_2(feature_1, feature_2)
        correlation_results[layer_1_index, :, :] = layer_correlations.cpu()
        co_occurrence_results[layer_1_index, :, :] = layer_co_occurrences.cpu()
# %%
# Clamp values below threshold to 0
correlation_results = correlation_results.clamp(min=similarity_threshold)

# %%
