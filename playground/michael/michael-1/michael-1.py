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
import requests


# %%
# Config
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
# Load Model and SAEs
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# blocks = list(range(len(model.blocks)))
blocks = [6, 7, 8]
saes = []
for block in tqdm(blocks):
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=f"blocks.{block}.hook_resid_pre", device=device)
    saes.append(sae)


# %%
# Perform similarity analysis for subsequent layers

# Define correlation function
def cosine_similarity(tensor1, tensor2):
    return torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)


# For simplicity, only consider first X features in each layer
number_of_features = 500

# Init similarity graph
layers = list(enumerate(blocks))
graph = nx.Graph()
graph.add_nodes_from([(f'{layer}_{feature}', {'layer': layer}) for _, layer in layers for feature in range(number_of_features)])

# For each pair of subsequent layers, look at each pair of features from different
# layers, calculate their similarity and add an edge in the similarity graph if it
# exceeds the threshold value
similarity_threshold = 0.5

results = []
for (index_1, layer_1), (index_2, layer_2) in zip(layers, layers[1:]):
    print(f'Computing feature similarities between layers {layer_1} and {layer_2}...')

    layer_1_feature_directions, layer_2_feature_directions = saes[index_1].W_dec, saes[index_2].W_dec

    layer_results = []
    for index_1, feature_1 in tqdm(enumerate(layer_1_feature_directions[:number_of_features]), total=number_of_features):
        feature_1_results = []
        for index_2, feature_2 in enumerate(layer_2_feature_directions[:number_of_features]):
            similarity = cosine_similarity(feature_1, feature_2).cpu().item()
            if similarity >= similarity_threshold:
                graph.add_edge(f'{layer_1}_{index_1}', f'{layer_2}_{index_2}', sim=similarity)

            feature_1_results.append(similarity)

        layer_results.append(feature_1_results)

    results.append(np.array(layer_results))


# %%
# Draw graph showing similaritys above threshold
nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key='layer'), node_size=10)


# %%
# %%
# Look up semantic information about high-similarity feature pairs on Neuronpedia
def top_k_indices(arr, k):
    # Flatten the array
    flat = arr.flatten()

    # Get the indices of the k largest elements
    indices = np.argpartition(flat, -k)[-k:]

    # Sort these indices
    sorted_indices = indices[np.argsort(-flat[indices])]

    # Convert flat indices to 2D indices
    unraveled_indices = np.unravel_index(sorted_indices, arr.shape)

    # Zip the unraveled indices to get a list of tuples
    top_k_indices = list(zip(*unraveled_indices))

    return top_k_indices


for (index_1, layer_1), (index_2, layer_2), layer_results in list(zip(layers, layers[1:], results)):
    top_feature_pairs = top_k_indices(np.nan_to_num(layer_results), 10)

    print(f'Top feature pairs between layers {layer_1} and {layer_2}:')
    for feature_1, feature_2 in top_feature_pairs:
        res_1 = requests.get(f'https://www.neuronpedia.org/api/feature/gpt2-small/{index_1}-res-jb/{feature_1}')
        res_2 = requests.get(f'https://www.neuronpedia.org/api/feature/gpt2-small/{index_2}-res-jb/{feature_2}')

        explanation_1 = res_1.json()['explanations'][0]['description'] if len(res_1.json()['explanations']) > 0 else None
        explanation_2 = res_2.json()['explanations'][0]['description'] if len(res_2.json()['explanations']) > 0 else None
        print(f'Similarity of {layer_results[feature_1][feature_2]:.4f} found between SAE features {layer_1}_{feature_1} ({explanation_1}) and {layer_2}_{feature_2} ({explanation_2}).')

    print()


# %%
# Plot 2D projection of SAE feature directions
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

feature_directions = [saes[index].W_dec for index, layer in layers]

# Combine the tensors
combined_feature_directions = torch.cat(feature_directions, dim=0)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(combined_feature_directions.cpu().detach().numpy())

# Split the principal components back into the original two sets
pcs = np.split(principal_components, len(layers))

# Plotting
plt.figure(figsize=(10, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))
for (index, layer), pc in zip(layers, pcs):
    plt.scatter(pc[:, 0], pc[:, 1], c=colors[index], label=f'Layer {layer}', alpha=0.3)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Feature directions')
plt.legend()
plt.show()
