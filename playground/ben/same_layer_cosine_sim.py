# %%
import re
from sae_lens import SAE
from tqdm import tqdm
import torch
import torch as t
import typing
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import List, Dict
import torch as t
import torch
import numpy as np
import typing
import einops
from sae_lens import SAE
from tqdm import tqdm
from functools import partial
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from itertools import combinations
# %%
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"
num_layers = 12
d_model = 768
prepend_bos = True
num_toks_per_row = 128
num_rows = 512
batch_size = 32
# %%
def create_id_to_sae() -> typing.Dict[str, SAE]:
    print("Loading SAEs")
    sae_id_to_sae = {}
    for layer in tqdm(list(range(num_layers))):
        sae_id = f"blocks.{layer}.hook_resid_pre"
        sae, _, _ = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=sae_id,
            device=device
        )
        sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
        sae_id_to_sae[sae_id] = sae
    return sae_id_to_sae
id_to_sae = create_id_to_sae()
# %%
layer_idx = 4
feat_idxes = [17018, 6551, 8314]
w_dec = id_to_sae[f"blocks.{layer_idx}.hook_resid_pre"].W_dec
for combo in combinations(feat_idxes, 2):
    feat1, feat2 = combo
    print(f"Layer {layer_idx}, Feature {feat1} vs Feature {feat2}")
    feat1_w = w_dec[feat1]
    feat2_w = w_dec[feat2]
    print(f"Cosine Similarity: {t.nn.functional.cosine_similarity(feat1_w, feat2_w, dim=0)}")
# %%
