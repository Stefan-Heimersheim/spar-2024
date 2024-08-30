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
from src import D_SAE


# %%
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"
num_layers = 12
d_model = 768
d_sae = 24576
prepend_bos = True
num_toks_per_row = 128
num_rows = 131072
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
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)
# %%
def load_data() -> DataLoader:
    dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=num_toks_per_row,
        add_bos_token=prepend_bos,
    )
    tokens = token_dataset['tokens'][:num_rows]
    print(f"Using {num_rows * num_toks_per_row} tokens")
    return DataLoader(tokens, batch_size=batch_size, shuffle=False)
data_loader = load_data()

# %%
projection_sums = t.zeros(num_layers-1, d_sae).to(device)
counts = t.zeros(num_layers-1, d_sae).to(device)
mask = t.zeros(batch_size, num_toks_per_row, d_sae).to(device)
max_activations_ten_percent = 0.1 * t.Tensor(np.load("artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz")['arr_0']).to(device)

# %%
# Compile the regex pattern once
PATTERN = re.compile(r'^blocks\.(\d+)\.hook_resid_pre$')

def is_any_resid_after_0(input_string):
    match = PATTERN.match(input_string)
    if match:
        # Extract the number and check if it's greater than 0
        num = int(match.group(1))
        return num > 0
    return False
    
def decrement_block_number(input_string):
    match = PATTERN.match(input_string)
    if match:
        number = int(match.group(1))
        new_number = number - 1
        if new_number < 0:
            return None  # Or you could raise an exception here
        return f"blocks.{new_number}.hook_resid_pre"
    return None

def set_mask(activations: t.Tensor, hook: HookPoint):
    global mask
    curr_layer_sae: SAE = id_to_sae[hook.name]
    sae_feats = curr_layer_sae.encode(activations)
    curr_layer_ten_percent_max_acts = max_activations_ten_percent[hook.layer()]
    mask = sae_feats > curr_layer_ten_percent_max_acts
    return activations

def compute_projections(activations: t.Tensor, hook: HookPoint):
    global projection_sums
    global counts
    prev_layer_sae: SAE = id_to_sae[decrement_block_number(hook.name)]
    curr_layer_sae: SAE = id_to_sae[hook.name]
    curr_layer_sae_error = activations - curr_layer_sae(activations)
    dot_product = einops.einsum(
        curr_layer_sae_error, prev_layer_sae.W_dec,
        "batch seq resid, sae resid -> batch seq sae",
    )
    decoder_norms = t.norm(prev_layer_sae.W_dec, dim=1)
    projections = dot_product / decoder_norms
    projections_plus_bias = projections + prev_layer_sae.b_dec
    masked_projections = projections * mask
    projection_sums[hook.layer()-1, :] += masked_projections.sum(dim=(0, 1))
    counts[hook.layer()-1, :] += mask.sum(dim=(0, 1))
    return activations
  
# %%
# following the logic described here https://spar2024.slack.com/archives/C077BCY4JTT/p1724199817182019
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.reset_hooks()
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                # compute projections for all layers after 0 (the mask was set for the previous layers)
                (
                    is_any_resid_after_0,
                    compute_projections,
                ),
                # set the mask for all layers, in preparation for the next layer to use it
                (
                    lambda name: name.endswith("hook_resid_pre"),
                    set_mask,
                ),
            ]
        )
# %%

np.savez(
    f"artefacts/projected_errs/with_bias__total_num_toks__{num_rows * num_toks_per_row}",
    projection_sums=projection_sums.cpu().numpy(),
    counts=counts.cpu().numpy()
)