# Given some correlation data in /artefacts, this script will show the degree to which it represents a causal relationship

# https://spar2024.slack.com/archives/C0794GNT8KS/p1721407323723129?thread_ts=1721403009.299289&cid=C0794GNT8KS
# %%
from src.pipeline_helpers import load_data2, get_device
import torch as t
import torch
import numpy as np
from transformer_lens import HookedTransformer
device = get_device()
# %%

# %%

# TODO: make this real
num_layers, num_features_per_layer = 2, 100
interaction_data = np.random.rand(num_layers, num_features_per_layer, num_features_per_layer)

# TODO: parameterize this into __main__
num_feature_pairs_to_test = 20
num_tokens_for_computing_ablation = 3

# %%
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

data = load_data2(model)