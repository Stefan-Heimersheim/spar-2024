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
import torch as t
import torch
import numpy as np
import typing
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
activation_sums = t.zeros(num_layers, d_sae).to(device)
counts = t.zeros(num_layers, d_sae).to(device)
max_activations_ten_percent = 0.1 * t.Tensor(np.load("artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz")['arr_0']).to(device)

# %%
def aggregate(activations: t.Tensor, hook: HookPoint):
    global activation_sums
    global counts
    curr_layer_sae: SAE = id_to_sae[hook.name]
    sae_feats = curr_layer_sae.encode(activations)
    curr_layer_max_feats = max_activations_ten_percent[hook.layer()]
    mask = sae_feats > curr_layer_max_feats
    masked_feats = sae_feats * mask
    activation_sums[hook.layer(), :] += masked_feats.sum(dim=(0, 1))
    counts[hook.layer(), :] += mask.sum(dim=(0, 1))
    return activations
  
# %%
# following the logic described here https://spar2024.slack.com/archives/C077BCY4JTT/p1724199817182019
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.reset_hooks()
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    lambda name: name.endswith("hook_resid_pre"),
                    aggregate,
                ),
            ]
        )
# %%
np.savez(
    f"artefacts/projected_errs/avg_masked_feats__total_num_toks__{num_rows * num_toks_per_row}",
    activation_sums=activation_sums.cpu().numpy(),
    counts=counts.cpu().numpy()
)