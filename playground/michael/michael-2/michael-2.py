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
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.clustering import MutualInfoScore


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


# %%
# Object for batched pair-wise Pearson correlation computation
class BatchedPearson:
    def __init__(self, shape):
        """Calculates the pair-wise Pearson correlation of two tensors that are provided batch-wise.

        Args:
            shape (Size): Shape of the result.
        """
        self.count = 0

        self.sums_1 = torch.zeros(shape[0])
        self.sums_2 = torch.zeros(shape[1])

        self.sums_of_squares_1 = torch.zeros(shape[0])
        self.sums_of_squares_2 = torch.zeros(shape[1])

        self.sums_1_2 = torch.zeros(shape)

    def process(self, tensor_1, tensor_2):
        self.count += tensor_1.shape[-1]

        self.sums_1 += tensor_1.sum(dim=-1)
        self.sums_2 += tensor_2.sum(dim=-1)

        self.sums_of_squares_1 += (tensor_1 ** 2).sum(dim=-1)
        self.sums_of_squares_2 += (tensor_2 ** 2).sum(dim=-1)

        self.sums_1_2 += einops.einsum(tensor_1, tensor_2, 'f1 t, f2 t -> f1 f2')

    def finalize(self):
        means_1 = self.sums_1 / self.count
        means_2 = self.sums_2 / self.count

        # Compute the covariance and variances
        covariances = (self.sums_1_2 / self.count) - einops.einsum(means_1, means_2, 'f1, f2 -> f1 f2')

        variances_1 = (self.sums_of_squares_1 / self.count) - (means_1 ** 2)
        variances_2 = (self.sums_of_squares_2 / self.count) - (means_2 ** 2)

        stds_1 = torch.sqrt(variances_1).unsqueeze(1)
        stds_2 = torch.sqrt(variances_2).unsqueeze(0)

        # Compute the Pearson correlation coefficient
        correlations = covariances / stds_1 / stds_2

        return correlations


# Nested for loops with own calculation
def pearson_1(tensor_1, tensor_2):
    result = torch.empty(tensor_1.shape[0], tensor_2.shape[0])
    for index_1, feature_1 in enumerate(tensor_1):
        for index_2, feature_2 in enumerate(tensor_2):
            # Calculate the mean of each tensor
            mean_1 = torch.mean(feature_1)
            mean_2 = torch.mean(feature_2)

            # Calculate the standard deviation of each tensor
            std_1 = torch.std(feature_1, unbiased=False)
            std_2 = torch.std(feature_2, unbiased=False)

            # Calculate the covariance between the two tensors
            covariance = torch.mean((feature_1 - mean_1) * (feature_2 - mean_2))

            # Calculate the Pearson correlation coefficient
            result[index_1, index_2] = covariance / (std_1 * std_2)

    return result


# Nested for loops with torchmetrics
def pearson_2(tensor_1, tensor_2):
    pearson = PearsonCorrCoef()

    result = torch.empty(tensor_1.shape[0], tensor_2.shape[0])
    for index_1, feature_1 in enumerate(tensor_1):
        for index_2, feature_2 in enumerate(tensor_2):
            result[index_1, index_2] = pearson(feature_1, feature_2)

    return result


# Own calculation with broadcasting
def pearson_3(tensor_1, tensor_2):
    number_of_tokens = tensor_1.shape[-1]

    means_1 = tensor_1.mean(dim=-1, keepdim=True)
    means_2 = tensor_2.mean(dim=-1, keepdim=True)

    stds_1 = tensor_1.std(dim=-1, unbiased=False)
    stds_2 = tensor_2.std(dim=-1, unbiased=False)

    centered_1 = tensor_1 - means_1
    centered_2 = tensor_2 - means_2

    covariances = einops.einsum(centered_1,
                                centered_2,
                                'f1 t, f2 t -> f1 f2'
                                ) / number_of_tokens
    correlations = covariances / stds_1.unsqueeze(1) / stds_2.unsqueeze(0)

    return correlations


f1 = torch.maximum(torch.rand(10, 1000) - 0.1, torch.tensor([0]))
f2 = torch.maximum(torch.rand(20, 1000) - 0.1, torch.tensor([0]))

true_result = pearson_1(f1, f2)
# print(f'True Pearson: {true_result}')

batch_size = 100
loader_1 = DataLoader(f1.movedim(-1, 0), batch_size=batch_size)
loader_2 = DataLoader(f2.movedim(-1, 0), batch_size=batch_size)
aggregator = BatchedPearson((f1.shape[0], f2.shape[0]))

for input_1, input_2 in zip(loader_1, loader_2):
    aggregator.process(input_1.movedim(0, -1), input_2.movedim(0, -1))

our_result = aggregator.finalize()
print(f'Calculated Pearson: {our_result}')

print(f'{torch.allclose(true_result, our_result, atol=1e-6)=}')


# %%
# Object for batched pair-wise mutual information computation
class BatchedMutualInformation:
    def __init__(self, shape, lower_bound=0.0):
        """Calculates the pair-wise Pearson correlation of two tensors that are provided batch-wise.

        Args:
            shape (Size): Shape of the result.
        """
        self.count = 0

        self.count_0_0 = torch.zeros(shape)
        self.count_0_1 = torch.zeros(shape)
        self.count_1_0 = torch.zeros(shape)
        self.count_1_1 = torch.zeros(shape)

        self.lower_bound = lower_bound

    def process(self, tensor_1, tensor_2):
        active_1 = (tensor_1 > self.lower_bound).unsqueeze(dim=1)
        active_2 = (tensor_2 > self.lower_bound).unsqueeze(dim=0)

        self.count += tensor_1.shape[-1]

        self.count_0_0 += (~active_1 & ~active_2).sum(dim=-1)
        self.count_0_1 += (~active_1 & active_2).sum(dim=-1)
        self.count_1_0 += (active_1 & ~active_2).sum(dim=-1)
        self.count_1_1 += (active_1 & active_2).sum(dim=-1)

    def finalize(self):
        # Compute joint probabilities
        p00 = self.count_0_0 / self.count
        p01 = self.count_0_1 / self.count
        p10 = self.count_1_0 / self.count
        p11 = self.count_1_1 / self.count

        # Compute marginal probabilities
        px0 = p00 + p01
        px1 = p10 + p11
        py0 = p00 + p10
        py1 = p01 + p11

        # Calculate mutual information
        mi = torch.zeros_like(p00)
        for pxy, px, py in zip([p00, p01, p10, p11], [px0, px0, px1, px1], [py0, py1, py0, py1]):
            mi += (pxy * torch.log(pxy / (px * py))).nan_to_num()

        return mi


# Nested for loops with own calculation
def mutual_1(tensor_1, tensor_2, lower_bound=0.0):
    active_1 = tensor_1 > lower_bound
    active_2 = tensor_2 > lower_bound

    n_tokens = active_1.shape[-1]

    result = torch.empty(active_1.shape[0], active_2.shape[0])
    for index_1, feature_1 in enumerate(active_1):
        for index_2, feature_2 in enumerate(active_2):
            # Compute the joint probability table
            p00 = torch.sum((feature_1 == 0) & (feature_2 == 0)) / n_tokens
            p01 = torch.sum((feature_1 == 0) & (feature_2 == 1)) / n_tokens
            p10 = torch.sum((feature_1 == 1) & (feature_2 == 0)) / n_tokens
            p11 = torch.sum((feature_1 == 1) & (feature_2 == 1)) / n_tokens

            # Compute marginal probabilities
            px0 = p00 + p01
            px1 = p10 + p11
            py0 = p00 + p10
            py1 = p01 + p11

            # Calculate mutual information
            mi = 0.0
            for pxy, px, py in zip([p00, p01, p10, p11], [px0, px0, px1, px1], [py0, py1, py0, py1]):
                if pxy > 0:
                    mi += pxy * torch.log(pxy / (px * py))

            result[index_1, index_2] = mi

    return result


# Nested for loops with torchmetrics
def mutual_2(tensor_1, tensor_2, lower_bound=0.0):
    active_1 = tensor_1 > lower_bound
    active_2 = tensor_2 > lower_bound

    mutual = MutualInfoScore()

    result = torch.empty(active_1.shape[0], active_2.shape[0])
    for index_1, feature_1 in enumerate(active_1):
        for index_2, feature_2 in enumerate(active_2):
            result[index_1, index_2] = mutual(feature_1, feature_2)

    return result


# Own calculation with broadcasting
def mutual_3(tensor_1, tensor_2, lower_bound=0.0):
    active_1 = (tensor_1 > lower_bound).unsqueeze(dim=1)
    active_2 = (tensor_2 > lower_bound).unsqueeze(dim=0)

    n_tokens = active_1.shape[-1]

    # Compute joint probabilities
    p00 = (~active_1 & ~active_2).sum(dim=-1) / n_tokens
    p01 = (~active_1 & active_2).sum(dim=-1) / n_tokens
    p10 = (active_1 & ~active_2).sum(dim=-1) / n_tokens
    p11 = (active_1 & active_2).sum(dim=-1) / n_tokens

    # Compute marginal probabilities
    px0 = p00 + p01
    px1 = p10 + p11
    py0 = p00 + p10
    py1 = p01 + p11

    # Calculate mutual information
    mi = torch.zeros(tensor_1.shape[0], tensor_2.shape[0])
    for pxy, px, py in zip([p00, p01, p10, p11], [px0, px0, px1, px1], [py0, py1, py0, py1]):
        mi += (pxy * torch.log(pxy / (px * py))).nan_to_num()

    return mi


f1 = torch.maximum(torch.rand(10, 1000) - 0.1, torch.tensor([0]))
f2 = torch.maximum(torch.rand(20, 1000) - 0.1, torch.tensor([0]))

true_result = mutual_1(f1, f2)
print(f'True Pearson: {true_result}')

batch_size = 100
loader_1 = DataLoader(f1.movedim(-1, 0), batch_size=batch_size)
loader_2 = DataLoader(f2.movedim(-1, 0), batch_size=batch_size)
aggregator = BatchedMutualInformation((f1.shape[0], f2.shape[0]))

for input_1, input_2 in zip(loader_1, loader_2):
    aggregator.process(input_1.movedim(0, -1), input_2.movedim(0, -1))

our_result = aggregator.finalize()
print(f'Calculated Pearson: {our_result}')

print(f'{torch.allclose(true_result, our_result, atol=2e-6)=}')
