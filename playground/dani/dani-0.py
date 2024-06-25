# %%
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
