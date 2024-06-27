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
# - Compute co-occurrence in batches

class BatchedCooccurrence:
    def __init__(self, shape, lower_bound=0.0, masked=True, device='cpu'):
        """Calculates the pair-wise co-occurrence of two 2d tensors that are provided batch-wise.

        Args:
            shape (Size): Shape of the result.
            lower_bound (float, optional): Lower bound for activation. Defaults to 0.0.
            masked (bool, optional): If true, only consider elements where at least one of the two tensors is active. Defaults to True.
        """
        self.count = torch.zeros(shape).to(device) if masked else 0
        self.sums = torch.zeros(shape).to(device)

        self.lower_bound = lower_bound
        self.masked = masked

    def process(self, tensor_1, tensor_2):
        active_1 = tensor_1 > self.lower_bound
        active_2 = tensor_2 > self.lower_bound

        if not self.masked:
            self.count += tensor_1.shape[-1]

        for index_1, feature_1 in enumerate(active_1):
            print('.', end='')
            for index_2, feature_2 in enumerate(active_2):
                if self.masked:
                    self.count[index_1, index_2] += (feature_1 | feature_2).sum()
                    self.sums[index_1, index_2] += (feature_1 & feature_2).sum()
                else:
                    self.sums[index_1, index_2] += (feature_1 == feature_2).sum()

    def finalize(self):
        return (self.sums / self.count)


batch_size = 32
data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

sae_features = []

co_occurrences = BatchedCooccurrence((10, 24576), device=device)
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        # Get model activations
        print(1)
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
        print(2)
        # Get and transform SAE feature activations
        feature_acts = torch.stack([sae.encode(cache[sae.cfg.hook_name]) for sae in saes])
        feature_acts = einops.rearrange(feature_acts, 'layers batch seq features -> layers features (batch seq)')

        # Delete cache to save memory
        del cache

        print(3)
        co_occurrences.process(feature_acts[0, :10], feature_acts[1, :])
        print(4)

    result = co_occurrences.finalize()

print(result.shape)


# %%
result.max(dim=-1)


# %%
# Compute variance of 1d feature vector chunk-wise
f = torch.rand(1000) * 10

print(f'True mean: {f.mean():.4f}')
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

print(f'True mean: {f.mean(dim=-1)}')
print(f'True variance: {f.var(dim=-1)}')


def update(existingAggregate, newValues):
    (count, mean, M2) = existingAggregate

    count += newValues.shape[-1]
    delta = newValues - mean.unsqueeze(dim=-1)

    mean += torch.sum(delta / count.unsqueeze(dim=-1), dim=1)
    delta2 = newValues - mean.unsqueeze(dim=-1)
    M2 += torch.sum(delta * delta2, dim=1)

    return (count, mean, M2)


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
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


# %%
# Compute variance of 3d feature vector chunk-wise
f = torch.rand(5, 12, 1000) * 10

print(f'True mean: {f.mean(dim=-1)}')
print(f'True variance: {f.var(dim=-1)}')


def update(existingAggregate, newValues):
    (count, mean, M2) = existingAggregate

    count += newValues.shape[-1]
    delta = newValues - mean.unsqueeze(dim=-1)

    mean += torch.sum(delta / count.unsqueeze(dim=-1), dim=-1)
    delta2 = newValues - mean.unsqueeze(dim=-1)
    M2 += torch.sum(delta * delta2, dim=-1)

    return (count, mean, M2)


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance) = (mean, M2 / (count - 1))

    mean[count < 2] = torch.nan
    variance[count < 2] = torch.nan

    return (mean, variance)


batch_size = 100
loader = DataLoader(f.T, batch_size=batch_size)

agg = torch.zeros(3, *f.shape[:-1])
for feature in loader:
    agg = update(agg, feature.T)

mean, variance = finalize(agg)

print(f'Calculated mean: {mean}')
print(f'Calculated variance: {variance}')


# %%
# Object for batched mean and variance computation
class BatchedStatistics:
    def __init__(self, shape):
        self.count = torch.zeros(shape)
        self.mean = torch.zeros(shape)
        self.m2 = torch.zeros(shape)

    def process(self, input):
        self.count += input.shape[-1]
        delta = input - self.mean.unsqueeze(dim=-1)

        self.mean += torch.sum(delta / self.count.unsqueeze(dim=-1), dim=-1)
        delta2 = input - self.mean.unsqueeze(dim=-1)
        self.m2 += torch.sum(delta * delta2, dim=-1)

    def finalize(self):
        mean, var = (self.mean, self.m2 / (self.count - 1))

        mean[self.count < 2] = torch.nan
        var[self.count < 2] = torch.nan

        return mean, var


f = torch.rand(5, 12, 1000) * 10

print(f'True mean: {f.mean(dim=-1)}')
print(f'True variance: {f.var(dim=-1)}')

batch_size = 100
loader = DataLoader(f.movedim(-1, 0), batch_size=batch_size)
stats = BatchedStatistics(f.shape[:-1])

for input in loader:
    stats.process(input.movedim(0, -1))

mean, variance = stats.finalize()

print(f'Calculated mean: {mean}')
print(f'Calculated variance: {variance}')


# %%
# Object for batched co-occurrence computation
class BatchedCooccurrence:
    def __init__(self, shape, lower_bound=0.0, masked=True):
        """Calculates the co-occurrence of two tensors that are provided batch-wise.

        Args:
            shape (Size): Shape of the tensors, excluding the dimension to iterate over.
            lower_bound (float, optional): Lower bound for activation. Defaults to 0.0.
            masked (bool, optional): If true, only consider elements where at least one of the two tensors is active. Defaults to True.
        """
        self.count = torch.zeros(shape)
        self.sum = torch.zeros(shape)

        self.lower_bound = lower_bound
        self.masked = masked

    def process(self, tensor_1, tensor_2):
        active_1 = tensor_1 > self.lower_bound
        active_2 = tensor_2 > self.lower_bound

        if self.masked:
            self.count += (active_1 | active_2).sum(dim=-1)
            self.sum += (active_1 & active_2).sum(dim=-1)
        else:
            self.count += tensor_1.shape[-1]
            self.sum += (active_1 == active_2).float().sum(dim=-1)

    def finalize(self):
        return self.sum / self.count


def co_occurrence(tensor_1, tensor_2, lower_bound=0.0, masked=True):
    active_1 = tensor_1 > lower_bound
    active_2 = tensor_2 > lower_bound

    if masked:
        return ((active_1 & active_2).sum(dim=-1) / (active_1 | active_2).sum(dim=-1))
    else:
        return (active_1 == active_2).float().mean(dim=-1)


f1 = torch.maximum(torch.rand(5, 12, 1000) - 0.9, torch.tensor([0]))
f2 = torch.maximum(torch.rand(5, 12, 1000) - 0.9, torch.tensor([0]))

lower_bound = 0.0
masked = False

print(f'True co_occurrence: {co_occurrence(f1, f2, lower_bound=lower_bound, masked=masked)}')

batch_size = 100
loader_1 = DataLoader(f1.movedim(-1, 0), batch_size=batch_size)
loader_2 = DataLoader(f2.movedim(-1, 0), batch_size=batch_size)
cooc = BatchedCooccurrence(f.shape[:-1], lower_bound=lower_bound, masked=masked)

for input_1, input_2 in zip(loader_1, loader_2):
    cooc.process(input_1.movedim(0, -1), input_2.movedim(0, -1))

print(f'Calculated co_occurrence: {cooc.finalize()}')


# %%
# Object for batched pair-wise co-occurrence computation
class BatchedCooccurrence:
    def __init__(self, shape, lower_bound=0.0, masked=True):
        """Calculates the co-occurrence of two tensors that are provided batch-wise.

        Args:
            shape (Size): Shape of the tensors, excluding the dimension to iterate over.
            lower_bound (float, optional): Lower bound for activation. Defaults to 0.0.
            masked (bool, optional): If true, only consider elements where at least one of the two tensors is active. Defaults to True.
        """
        self.count = torch.zeros(shape) if masked else 0
        self.sums = torch.zeros(shape)

        self.lower_bound = lower_bound
        self.masked = masked

    def process(self, tensor_1, tensor_2):
        active_1 = tensor_1 > self.lower_bound
        active_2 = tensor_2 > self.lower_bound

        if not self.masked:
            self.count += tensor_1.shape[-1]

        for index_1, feature_1 in enumerate(active_1):
            for index_2, feature_2 in enumerate(active_2):
                if self.masked:
                    self.count[index_1, index_2] += (feature_1 | feature_2).sum()
                    self.sums[index_1, index_2] += (feature_1 & feature_2).sum()
                else:
                    self.sums[index_1, index_2] += (feature_1 == feature_2).sum()

    def finalize(self):
        return self.sums / self.count


def co_occurrence(tensor_1, tensor_2, lower_bound=0.0, masked=True):
    active_1 = tensor_1 > lower_bound
    active_2 = tensor_2 > lower_bound

    result = torch.empty(tensor_1.shape[0], tensor_2.shape[0])

    for index_1, feature_1 in enumerate(active_1):
        for index_2, feature_2 in enumerate(active_2):
            if masked:
                result[index_1, index_2] = ((feature_1 & feature_2).sum() / (feature_1 | feature_2).sum())
            else:
                result[index_1, index_2] = (feature_1 == feature_2).float().mean()

    return result


f1 = torch.maximum(torch.rand(10, 1000) - 0.1, torch.tensor([0]))
f2 = torch.maximum(torch.rand(20, 1000) - 0.1, torch.tensor([0]))

lower_bound = 0.8
masked = True

true_result = co_occurrence(f1, f2, lower_bound=lower_bound, masked=masked)
# print(f'True co_occurrence: {true_result}')

batch_size = 100
loader_1 = DataLoader(f1.movedim(-1, 0), batch_size=batch_size)
loader_2 = DataLoader(f2.movedim(-1, 0), batch_size=batch_size)
cooc = BatchedCooccurrence((f1.shape[0], f2.shape[0]), lower_bound=lower_bound, masked=masked)

for input_1, input_2 in zip(loader_1, loader_2):
    cooc.process(input_1.movedim(0, -1), input_2.movedim(0, -1))

our_result = cooc.finalize()
# print(f'Calculated co_occurrence: {our_result}')

print(f'{true_result.equal(our_result)=}')
