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
from torch.utils.data import DataLoader


# %%
# Config
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
# Load Model, SAEs and data
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

blocks = [6, 7]
saes = []
for block in tqdm(blocks):
    sae, _, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=f"blocks.{block}.hook_resid_pre", device=device)
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    saes.append(sae)

# These hyperparameters are used to pre-process the data
context_size = saes[0].cfg.context_size
prepend_bos = saes[0].cfg.prepend_bos

dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=context_size,
    add_bos_token=prepend_bos,
)

tokens = token_dataset['tokens']


# %%
# OPTIONAL: Reduce dataset for faster experimentation
tokens = tokens[:64]


# %%
# Run model in batches:
# - Cache activations
# - Compute SAE feature activations
# - Compute correlation in chunks

batch_size = 32
data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

sae_features = []

means = torch.empty
dot_prods = torch.zeros(4, 24576).to('mps')
sums = torch.zeros(len(saes), 24576).to('mps')
sums_of_squares = torch.zeros(len(saes), 24576).to('mps')
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        # Get model activations
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)

        # Get and transform SAE feature activations
        feature_acts = torch.stack([sae.encode(cache[sae.cfg.hook_name]) for sae in saes])
        feature_acts = einops.rearrange(feature_acts, 'layers batch seq features -> layers features (batch seq)')

        # Delete cache to save memory
        del cache

        sums += feature_acts.sum(dim=-1)
        sums_of_squares += (feature_acts ** 2).sum(dim=-1)

        dot_prods += einops.einsum(feature_acts[0, :4], feature_acts[1, :], 'f1 tokens, f2 tokens -> f1 f2')

    number_of_samples = len(tokens)

    # Compute the means and variances
    means = sums / number_of_samples
    variances = (sums_of_squares / number_of_samples) - (means ** 2)

    # Compute the covariance and variances
    covariances = (dot_prods / number_of_samples) - (means[0] * means[1])

    # Compute the Pearson correlation coefficient
    correlations = covariances / (torch.sqrt(variances[0]) * torch.sqrt(variances[1]))


# %%
# Compute Pearson correlation of feature vectors chunk-wsye
from torchmetrics.regression import PearsonCorrCoef

f_1 = torch.rand(1000) * 10
f_2 = torch.rand(1000) * 10

print(f'True correlation: {PearsonCorrCoef()(f_1, f_2):.4f}')

batch_size = 100
loader_1 = DataLoader(f_1, batch_size=batch_size)
loader_2 = DataLoader(f_2, batch_size=batch_size)

for feature_1, feature_2 in zip(loader_1, loader_2):
    print(0)


# %%
# Compute variance of 1d feature vector chunk-wise
f = torch.rand(1000) * 10

print(f'True variance: {f.var():.4f}')


def update(existingAggregate, newValues):
    (count, mean, M2) = existingAggregate
    count += len(newValues)
    delta = newValues - mean
    mean += torch.sum(delta / count)
    delta2 = newValues - mean
    M2 += torch.sum(delta * delta2)

    return torch.tensor([count, mean, M2])


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance) = (mean, M2/(count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance)


batch_size = 100
loader = DataLoader(f, batch_size=batch_size)

agg = torch.zeros(3)
for feature in loader:
    agg = update(agg, feature)

finalize(agg)

# %%
# Compute variance of 2d feature vector chunk-wise
f = torch.rand(5, 1000) * 10

print(f'True variance: {f.var(dim=-1)}')


def update(existingAggregate, newValues):
    (count, mean, M2) = existingAggregate

    count += len(newValues)
    print(count)
    delta = newValues - mean.unsqueeze(dim=-1)
    mean += torch.sum(delta / count.unsqueeze(dim=-1))
    delta2 = newValues - mean.unsqueeze(dim=-1)
    M2 += torch.sum(delta * delta2)

    return (count, mean, M2)


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    print(count, mean, M2)
    (mean, variance) = (mean, M2 / (count - 1))

    mean[count < 2] = torch.nan
    variance[count < 2] = torch.nan

    return (mean, variance)


batch_size = 100
loader = DataLoader(f.T, batch_size=batch_size)

agg = torch.zeros(3, f.shape[0])
for feature in loader:
    agg = update(agg, feature.T)

finalize(agg)