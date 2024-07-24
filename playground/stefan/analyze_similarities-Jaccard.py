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
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "src"))

from similarity_helpers import get_filename, load_correlation_data


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


# %%
# Load (clamped) similarity matrix
measure_name = "jaccard_similarity"
REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")
folder = f"{REPO_ROOT}/artefacts/similarity_measures/{measure_name}"
filename = get_filename(measure_name, 0.0, 0.1, "1M")
similarities = np.load(f"{folder}/{filename}.npz")["arr_0"]

# %%
# Stefan
for downstream in range(2000, 3000, 1):
    layer = 6
    sims = similarities[layer][:, downstream]
    layer_in = layer
    layer_out = layer + 1
    if max(sims) < 0.9:
        continue
    print("Similarity between layer", layer_in, "and layer", layer_out)
    downstream_explanation = get_explanation(layer_out, downstream)
    for upstream in np.argsort(sims)[::-1][:5]:
        upstream_explanation = get_explanation(layer_in, upstream)
        print(
            f"Feature {layer_in}.{upstream} -> {layer_out}.{downstream} ({sims[upstream]})"
        )
        print(f"Explanation {upstream_explanation} -> {downstream_explanation}")
