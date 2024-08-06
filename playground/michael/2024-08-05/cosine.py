# %%
# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
import torch
import einops
import numpy as np
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from pipeline_helpers import load_model_and_saes
from similarity_helpers import clamp_low_values, save_compressed


# OPTIONAL: Check if the correct GPU is visible
print(torch.cuda.device_count())  # Should print 1
print(torch.cuda.current_device())  # Should print 0 since it's the first visible device
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)

# %%
n_layers = model.cfg.n_layers
d_sae = saes[0].cfg.d_sae

similarities = torch.empty(n_layers - 1, d_sae, d_sae)
for layer, sae_1, sae_2 in tqdm(zip(range(n_layers), saes, saes[1:]), total=n_layers - 1):
    similarities[layer] = sae_1.W_dec @ sae_2.W_dec.T

similarities = similarities.detach()


# %%
clamping_threshold = 0.4
clamp_low_values(similarities, clamping_threshold)
similarities.count_nonzero()

# %%
folder = f'../../../artefacts/similarity_measures/cosine_similarity'
save_compressed(similarities, f'{folder}/res_jb_sae_feature_similarity_cosine_similarity_{clamping_threshold:.1f}')

# %%
