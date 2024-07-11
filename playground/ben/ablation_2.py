## https://spar2024.slack.com/archives/C0794GNT8KS/p1720559352313959?thread_ts=1720550352.237039&cid=C0794GNT8KS

"""
Run ablation of feature in layer 1 and record all features in layer 2 at almost no additional cost -->
getting lots more ablation scores at once

1. implement for all layer 2 features
2. pick 3 random layer 1 features
3. collect ablation acts for each layer 1 feature
"""

# %%
import torch as t
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
import einops
from dataclasses import dataclass
from typing import List
from functools import partial
import jaxtyping as jt
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader

if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"


# %%
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
sae_id_to_sae = {}
for layer in tqdm(list(range(model.cfg.n_layers))):
    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=sae_id,
        device=device
    )
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    sae_id_to_sae[sae_id] = sae
    

# %%
# These hyperparameters are used to pre-process the data
pre_0_sae_id = "blocks.0.hook_resid_pre"
pre_0_sae = sae_id_to_sae[pre_0_sae_id]
context_size = pre_0_sae.cfg.context_size
prepend_bos = pre_0_sae.cfg.prepend_bos
d_sae = pre_0_sae.cfg.d_sae
batch_size = 32

dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)
token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=context_size,
    add_bos_token=prepend_bos,
)

# OPTIONAL: Reduce dataset for faster experimentation
num_of_sentences = 1024
tokens = token_dataset['tokens'][:num_of_sentences]
data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)
# %%
sae_errors = t.empty(batch_size, context_size, model.cfg.d_model)
second_layer_unablated_acts = t.empty(batch_size, context_size, d_sae)
second_layer_ablated_acts = t.empty(batch_size, context_size, d_sae)

def save_error_terms(activations: t.Tensor, hook: HookPoint):
    global sae_errors
    sae: SAE = sae_id_to_sae[hook.name]
    reconstructed_acts = sae(activations)
    sae_errors = activations - reconstructed_acts 
    return activations

def save_second_layer_acts(activations: t.Tensor, hook: HookPoint, ablated: bool):
    global second_layer_unablated_acts 
    global second_layer_ablated_acts 
    sae: SAE = sae_id_to_sae[hook.name]
    sae_feats = sae.encode(activations)
    if ablated:
        second_layer_ablated_acts = sae_feats
    else:
        second_layer_unablated_acts = sae_feats
    return activations

def ablate_and_reconstruct_with_errors(activations: t.Tensor, hook: HookPoint, feature_idx: int):
    global sae_errors
    sae: SAE = sae_id_to_sae[hook.name]
    sae_feats = sae.encode(activations)
    sae_feats[:,:,feature_idx] = 0    
    return sae.decode(sae_feats) + sae_errors # we assume that the hooks are called in the correct order s.t. these errors correspond to the same tokens

class DiffAgg:
    def __init__(self) -> None:
        self.sum_of_diffs = t.zeros(d_sae).to(device)
        self.sum_of_squared_diffs = t.zeros(d_sae).to(device)
        self.total_num_of_diffs = 0
        self.sum_of_masked_diffs = t.zeros(d_sae).to(device)
        self.total_num_of_masked_diffs = 0
        
        # self.max_unablated_acts = t.zeros(d_sae).to(device)
        # self.max_ablated_acts = t.zeros(d_sae).to(device)
    
    def process_global_second_layer_acts(self):
        diffs = second_layer_unablated_acts - second_layer_ablated_acts
        self.sum_of_diffs += t.sum(diffs, dim=(0, 1))
        self.sum_of_squared_diffs += t.sum(diffs.pow(2), dim=(0, 1))
        num_diffs = diffs.shape[0] * diffs.shape[1]
        self.total_num_of_diffs += num_diffs
        # self.max_unablated_acts = t.max(self.max_unablated_acts, second_layer_unablated_acts)
        # self.max_ablated_acts = t.max(self.max_ablated_acts, second_layer_ablated_acts)
        
    def finalize(self):
        self.means = self.sum_of_diffs / self.total_num_of_diffs
        self.variances = (self.sum_of_squared_diffs / self.total_num_of_diffs) - (self.means ** 2)
        self.std_devs = t.sqrt(self.variances)
        
first_layer_feat_idx = 10715
first_layer_idx = 0
count = 0
diff_agg = DiffAgg()
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        if count >= 1:
            break
        model.reset_hooks()
        # collect the unablated activations
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    f"blocks.{first_layer_idx}.hook_resid_pre", 
                    save_error_terms,
                ),
                (
                    f"blocks.{first_layer_idx+1}.hook_resid_pre", 
                    partial(save_second_layer_acts, ablated=False),
                )
            ]
        )
        # collect the activations after ablation
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    f"blocks.{first_layer_idx}.hook_resid_pre", 
                    partial(ablate_and_reconstruct_with_errors, feature_idx=first_layer_feat_idx)
                ),
                (
                    f"blocks.{first_layer_idx+1}.hook_resid_pre", 
                    partial(save_second_layer_acts, ablated=True)
                )
            ]
        )
        diff_agg.process_global_second_layer_acts()
        count += 1
diff_agg.finalize()
print(diff_agg.means)
print(diff_agg.std_devs)
print(diff_agg.max_unablated_acts)
print(diff_agg.max_ablated_acts)
## 
# 0.05463759% 

# %%
print(diff_agg.means)
"""
# ideal case - how do we know this is successful? lots of 0s

layer_2_unablated_act | layer_2_ablated_act
0 0
0 0
0 0
8 1
10 2
7 1
0 0
0 0
0 0
...

# stranger case - consistently weird impact on feature 2
0 0
0 0
0 0
10 15
7 4
8 14
9 3
10 5
0 0


# STEPS
1. filter down to ONLY the times when layer 2, unablated, was nonzero
2. measure mean diff and std dev of diff
-----
1. calculate MSE
2. explore results

"""
# %%
