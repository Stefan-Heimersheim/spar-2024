# %%
# Imports
import os
import torch
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.clustering import MutualInfoScore
from tqdm import trange
from colorama import init, Fore, Style

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from similarity_measures import JaccardSimilarityAggregator, PearsonCorrelationAggregator, ForwardImplicationAggregator, BackwardImplicationAggregator, MutualInformationAggregator

# Init colorama
init()

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
    
    jaccard = BinaryJaccardIndex()
    for feature_1 in range(n_features):
          for feature_2 in range(n_features):
                jaccard_similarity[feature_1, feature_2] = jaccard(active_1[feature_1], active_2[feature_2])

    return jaccard_similarity


def manual_pearson(activations, layer, lower_bound=0.0):
    n_features = activations.shape[1]

    pearson_correlation = torch.empty(n_features, n_features)

    activations_1 = activations[layer]
    activations_2 = activations[layer + 1]
    
    pearson = PearsonCorrCoef()
    for feature_1 in range(n_features):
          for feature_2 in range(n_features):
                pearson_correlation[feature_1, feature_2] = pearson(activations_1[feature_1], activations_2[feature_2])

    return pearson_correlation


def manual_mutual(activations, layer, lower_bound=0.0):
    n_features = activations.shape[1]

    mutual_information = torch.empty(n_features, n_features)

    active_1 = activations[layer] > lower_bound
    active_2 = activations[layer + 1] > lower_bound
    
    for feature_1 in range(n_features):
          for feature_2 in range(n_features):
                mutual_information[feature_1, feature_2] = MutualInfoScore()(active_1[feature_1], active_2[feature_2])

    return mutual_information


def one_dim_forward_implication(A, B):
    count_A_and_B = torch.logical_and(A, B).sum()
    count_B = torch.sum(B)

    return count_A_and_B / count_B  # can be nan


def manual_forward(activations, layer, lower_bound=0.0):
    n_features = activations.shape[1]

    forward_implication = torch.empty(n_features, n_features)

    active_1 = activations[layer] > lower_bound
    active_2 = activations[layer + 1] > lower_bound
    
    for feature_1 in range(n_features):
          for feature_2 in range(n_features):
                forward_implication[feature_1, feature_2] = one_dim_forward_implication(active_1[feature_1], active_2[feature_2])

    return forward_implication


def one_dim_backward_implication(A, B):
    count_A_and_B = torch.logical_and(A, B).sum()
    count_A = torch.sum(A)

    return count_A_and_B / count_A  # can be nan


def manual_backward(activations, layer, lower_bound=0.0):
    n_features = activations.shape[1]

    backward_implication = torch.empty(n_features, n_features)

    active_1 = activations[layer] > lower_bound
    active_2 = activations[layer + 1] > lower_bound
    
    for feature_1 in range(n_features):
          for feature_2 in range(n_features):
                backward_implication[feature_1, feature_2] = one_dim_backward_implication(active_1[feature_1], active_2[feature_2])

    return backward_implication


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
    for layer in range(n_layers - 1):
        manual_jaccard_similarity = manual_jaccard(activations, layer)
        aggregated_jaccard_similarity = batched_loop(activations, layer, JaccardSimilarityAggregator)

        if torch.allclose(manual_jaccard_similarity, aggregated_jaccard_similarity, atol=1e-6):
            print(f'{Fore.GREEN}Passed Jaccard test for layers {layer}/{layer+1}.{Style.RESET_ALL}')
        else:
            print(f'{Fore.RED}Failed Jaccard test for layers {layer}/{layer+1}!{Style.RESET_ALL}')


def test_pearson(activations):
    for layer in range(n_layers - 1):
        manual_pearson_correlation = manual_pearson(activations, layer)
        aggregated_pearson_correlation = batched_loop(activations, layer, PearsonCorrelationAggregator)

        if torch.allclose(manual_pearson_correlation, aggregated_pearson_correlation, atol=1e-6):
            print(f'{Fore.GREEN}Passed Pearson test for layers {layer}/{layer+1}.{Style.RESET_ALL}')
        else:
            print(f'{Fore.RED}Failed Pearson test for layers {layer}/{layer+1}!{Style.RESET_ALL}')


def test_mutual(activations):
    for layer in range(n_layers - 1):
        manual_mutual_information = manual_mutual(activations, layer)
        aggregated_mutual_information = batched_loop(activations, layer, MutualInformationAggregator)

        if torch.allclose(manual_mutual_information, aggregated_mutual_information, atol=1e-6):
            print(f'{Fore.GREEN}Passed Mutual Information test for layers {layer}/{layer+1}.{Style.RESET_ALL}')
        else:
            max_error = (manual_mutual_information - aggregated_mutual_information).max().item()
            print(f'{Fore.RED}Failed Mutual Information test for layers {layer}/{layer+1} ({max_error})!{Style.RESET_ALL}')


def test_forward(activations):
    for layer in range(n_layers - 1):
        manual_forward_implication = manual_forward(activations, layer)
        aggregated_forward_implication = batched_loop(activations, layer, ForwardImplicationAggregator)

        if torch.allclose(manual_forward_implication, aggregated_forward_implication, atol=1e-6):
            print(f'{Fore.GREEN}Passed Forward Implication test for layers {layer}/{layer+1}.{Style.RESET_ALL}')
        else:
            max_error = (manual_forward_implication - aggregated_forward_implication).max().item()
            print(f'{Fore.RED}Failed Forward Implication test for layers {layer}/{layer+1} ({max_error})!{Style.RESET_ALL}')


def test_backward(activations):
    for layer in range(n_layers - 1):
        manual_backward_implication = manual_backward(activations, layer)
        aggregated_backward_implication = batched_loop(activations, layer, BackwardImplicationAggregator)

        if torch.allclose(manual_backward_implication, aggregated_backward_implication, atol=1e-6):
            print(f'{Fore.GREEN}Passed Backward Implication test for layers {layer}/{layer+1}.{Style.RESET_ALL}')
        else:
            max_error = (manual_backward_implication - aggregated_backward_implication).max().item()
            print(f'{Fore.RED}Failed Backward Implication test for layers {layer}/{layer+1} ({max_error})!{Style.RESET_ALL}')


test_jaccard(activations)
test_pearson(activations)
test_mutual(activations)
test_forward(activations)
test_backward(activations)

# %%
