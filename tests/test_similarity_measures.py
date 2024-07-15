# %%
# Imports
import os
import torch
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import trange

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from similarity_measures import JaccardSimilarityAggregator, PearsonCorrelationAggregator, ForwardImplicationAggregator, BackwardImplicationAggregator


# %%
# Define parameters
n_layers, d_sae, n_tokens = 12, 128, 1024

# %%
# Generate correlated sample data with shape (n_layers, d_sae, n_tokens)
activations = torch.rand(n_layers, d_sae, n_tokens) * 100 - 90

for layer in range(1, n_layers):
        activations[layer] = torch.maximum(activations[layer - 1] + torch.rand(d_sae, n_tokens) * 20 - 10, torch.tensor([0]))


# %%
# Define manual (looped) similarity functions
def manual_jaccard(activations, layer, lower_bound=0.0):
    n_features = activations.shape[1]

    jaccard_similarity = torch.empty(n_features, n_features)

    active_1 = activations[layer] > lower_bound
    active_2 = activations[layer + 1] > lower_bound
    
    for feature_1 in range(n_features):
          for feature_2 in range(n_features):
                jaccard_similarity[feature_1, feature_2] = BinaryJaccardIndex()(active_1[feature_1], active_2[feature_2])

    return jaccard_similarity


# %%
# Define batched loop
def batched_loop(activations, layer, aggregator_cls, batch_size=32):
    n_features = activations.shape[1]
    n_batches = activations.shape[-1] // batch_size 
    
    aggregator = aggregator_cls(layer, (n_features, n_features))
    for batch_index in range(n_batches):
        batch = activations[:, :, (batch_index * batch_size):((batch_index + 1) * batch_size)]

        aggregator.process(batch)

    return aggregator.finalize()

# %%
# Run manual and aggregated functions and compare results
def test_jaccard(activations):
    for layer in trange(n_layers - 1):
        manual_jaccard_similarity = manual_jaccard(activations, layer)
        aggregated_jaccard_similarity = batched_loop(activations, layer, JaccardSimilarityAggregator)

        if torch.allclose(manual_jaccard_similarity, aggregated_jaccard_similarity, atol=1e-6):
            print(f'Passed Jaccard test for layers {layer}/{layer+1}.')

test_jaccard(activations)

# TODO: Add tests for remaining similarity measures