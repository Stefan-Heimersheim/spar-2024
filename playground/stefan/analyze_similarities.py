# %%
# Imports
import concurrent.futures
import os
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
import torch
from sae_lens import SAE
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "src"))

from similarity_helpers import get_filename


def get_explanation(
    layer, feature, model_name="gpt2-small", sae_name="res-jb"
) -> Optional[str]:
    """Fetches a single explanation for a given layer and feature from Neuronpedia."""
    res = requests.get(
        f"https://www.neuronpedia.org/api/feature/{model_name}/{layer}-{sae_name}/{feature}"
    ).json()
    explanation = (
        res["explanations"][0]["description"] if len(res["explanations"]) > 0 else None
    )

    return explanation


# %% Load (clamped) similarity matrix
# measure_name = "jaccard_similarity"
measure_name = "pearson_correlation"
REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")
folder = f"{REPO_ROOT}/artefacts/similarity_measures/{measure_name}"
filename = get_filename(measure_name, 0.0, 0.1, "1M")
similarities = np.load(f"{folder}/{filename}.npz")["arr_0"]
# %% Get SAEs
saes = {}
device = torch.device("cpu")
for layer in range(12):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=sae_id)
    saes[layer] = sae

# %% Load model
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %% Run a forward pass, cache activations, gere
prompt = "Today, Marry and John went to the park. They"
logits, cache = model.run_with_cache(prompt)
# Get sae feature activations on
# %%
# Stefan
for downstream in range(2000, 3000, 1):
    layer = 6
    sims = similarities[layer][:, downstream]
    layer_in = layer
    layer_out = layer + 1
    if max(sims) < 0.9:
        continue
    # print("Similarity between layer", layer_in, "and layer", layer_out)
    downstream_explanation = get_explanation(layer_out, downstream)
    for upstream in np.argsort(sims)[::-1][:5]:
        upstream_explanation = get_explanation(layer_in, upstream)
        print(
            f"Feature {layer_in}.{upstream:05} -> {layer_out}.{downstream:05} ({sims[upstream]:.2f}): {upstream_explanation} -> {downstream_explanation}"
        )
# %% Calculate cosine sims between some of these
# 6.20933 -> 7.2003
l6 = saes[6].W_dec[20933].detach().cpu().numpy()
l7 = saes[7].W_dec[2003].detach().cpu().numpy()
print(
    "Cosine similarity between 6.20933 and 7.2003 is",
    np.dot(l6, l7) / (np.linalg.norm(l6) * np.linalg.norm(l7)),
)
# 6.23579 -> 7.2005
l6 = saes[6].W_dec[23579].detach().cpu().numpy()
l7 = saes[7].W_dec[2005].detach().cpu().numpy()
print(
    "Cosine similarity between 6.23579 and 7.2005 is",
    np.dot(l6, l7) / (np.linalg.norm(l6) * np.linalg.norm(l7)),
)
# %%
