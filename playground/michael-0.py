# %%
# Imports
import torch
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


# %%
# Config
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
# Load Model, SAEs and data
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# blocks = list(range(len(model.blocks)))
blocks = [6, 7, 8, 9]
saes = []
for block in tqdm(blocks):
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=f"blocks.{block}.hook_resid_pre", device=device)
    saes.append(sae)


# %%
dataset = load_dataset(path = "NeelNanda/pile-10k", split="train", streaming=False)

token_dataset = tokenize_and_concatenate(
    dataset= dataset, # type: ignore
    tokenizer = model.tokenizer, # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)


# %%
# Get activations and compute feature activations
for sae in saes:
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with torch.no_grad():
    batch_tokens = token_dataset[:32]["tokens"]
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
# Perform correlation analysis for subsequent layers

# Define correlation function
# TODO: Improve efficiency
def pearson_correlation(tensor1, tensor2):
    # Ensure the tensors have the same length
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    
    # Calculate the mean of each tensor
    mean1 = torch.mean(tensor1)
    mean2 = torch.mean(tensor2)
    
    # Calculate the standard deviation of each tensor
    std1 = torch.std(tensor1, unbiased=False)
    std2 = torch.std(tensor2, unbiased=False)
    
    # Calculate the covariance between the two tensors
    covariance = torch.mean((tensor1 - mean1) * (tensor2 - mean2))
    
    # Calculate the Pearson correlation coefficient
    correlation = covariance / (std1 * std2)
    
    return correlation.item()

# Rearrange feature activations
acts_per_feature = [einops.rearrange(layer_feature_acts, 'b p f -> f (b p)') for layer_feature_acts in feature_acts]

# For simplicity, only consider first X features in each layer
number_of_features = 100

# Init correlation graph
layers = list(enumerate(blocks))
graph = nx.Graph()
graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer}) for _, layer in layers for feature in range(number_of_features)])

# For each pair of subsequent layers, look at each pair of features from different layers, calculate their correlation
# and add an edge in the correlation graph if it exceeds the threshold value
correlation_threshold = 0.2

for (index_1, layer_1), (index_2, layer_2) in tqdm(list(zip(layers, layers[1:]))):
    print(f'Computing feature correlations between layers {layer_1} and {layer_2}...')

    layer_1_acts_per_feature, layer_2_acts_per_feature = acts_per_feature[index_1], acts_per_feature[index_2]
    
    for index_1, feature_1 in tqdm(enumerate(layer_1_acts_per_feature[:number_of_features]), total=number_of_features, leave=False):
        for index_2, feature_2 in enumerate(layer_2_acts_per_feature[:number_of_features]):
            correlation = pearson_correlation(feature_1, feature_2)
            if correlation >= correlation_threshold:
                graph.add_edge(f'{layer_1}_{index_1}', f'{layer_2}_{index_2}', corr=correlation)
    

# %%
nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key='layer'), node_size=10)