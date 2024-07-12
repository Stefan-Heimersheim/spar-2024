# This file contains the Aggregator classes for all similarity measures
# used in this project.

# %%
# Imports
from typing import Tuple
from jaxtyping import Float, Int
from torch import Tensor

import torch
import einops


# %%
class PearsonCorrelationAggregator:
    def __init__(self, layer: int, n_features: Tuple[int, int]):
        """Calculates the pair-wise Pearson correlation of two tensors that are
        provided batch-wise. All computations are done element-wise with einsum.

        Args:
            layer (int): First of the two subsequent layers for similarity computation.
            n_features (Tuple[int, int]): Number of features to include per layer.
        """
        self.layer = layer
        self.n_features = n_features

        # Init aggregator variables
        self.count = 0

        n_features_1, n_features_2 = n_features

        self.sums_1 = torch.zeros(n_features_1)
        self.sums_2 = torch.zeros(n_features_2)

        self.sums_of_squares_1 = torch.zeros(n_features_1)
        self.sums_of_squares_2 = torch.zeros(n_features_2)

        self.sums_1_2 = torch.zeros(n_features)

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        n_features_1, n_features_2 = self.n_features
        n_tokens = activations.shape[-1]
        
        activations_1 = activations[self.layer, :n_features_1, :]
        activations_2 = activations[self.layer + 1, :n_features_2, :]
        
        # Aggregate count, sums and sums of squares
        self.count += n_tokens

        self.sums_1 += activations_1.sum(dim=-1)
        self.sums_2 += activations_2.sum(dim=-1)

        self.sums_of_squares_1 += (activations_1 ** 2).sum(dim=-1)
        self.sums_of_squares_2 += (activations_2 ** 2).sum(dim=-1)

        self.sums_1_2 += einops.einsum(activations_1, activations_2, "n_features_1 n_tokens, n_features_2 n_tokens -> n_features_1 n_features_2")

    def finalize(self) -> Float[Tensor, 'n_features_1 n_features_2']:
        # Compute means
        means_1 = self.sums_1 / self.count
        means_2 = self.sums_2 / self.count

        # Compute the covariance, variances, and standard deviations
        covariances = (self.sums_1_2 / self.count) - einops.einsum(
            means_1, means_2, "n_features_1, n_features_2 -> n_features_1 n_features_2"
        )

        variances_1 = (self.sums_of_squares_1 / self.count) - (means_1 ** 2)
        variances_2 = (self.sums_of_squares_2 / self.count) - (means_2 ** 2)

        stds_1 = torch.sqrt(variances_1).unsqueeze(1)
        stds_2 = torch.sqrt(variances_2).unsqueeze(0)

        # Compute the Pearson correlation coefficients
        correlations = covariances / stds_1 / stds_2

        return correlations


class ForwardImplicationAggregator:
    def __init__(self, layer: int, n_features: Tuple[int, int], lower_bound=0.0):
        """Calculates the pair-wise forward implication of two tensors that are
        provided batch-wise. All computations are done element-wise with einsum.

        Args:
            layer (int): First of the two subsequent layers for similarity computation.
            n_features (Tuple[int, int]): Number of features to include per layer.
            lower_bound (float): Threshold to distinguish between active and inactive.
        """
        self.layer = layer
        self.n_features = n_features
        self.lower_bound = lower_bound

        n_features_1, n_features_2 = n_features

        self.counts = torch.zeros(n_features_1, n_features_2)
        self.sums = torch.zeros(n_features_1, n_features_2)

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        n_features_1, n_features_2 = self.n_features
        
        activations_1 = activations[self.layer, :n_features_1, :]
        activations_2 = activations[self.layer + 1, :n_features_2, :]

        active_1 = (activations_1 > self.lower_bound).float()
        active_2 = (activations_2 > self.lower_bound).float()

        self.counts += active_1.sum(dim=-1).unsqueeze(1)
        self.sums += einops.einsum(active_1, active_2, 'n_features_1 n_tokens, n_features_2 n_tokens -> n_features_1 n_features_2')

    def finalize(self) -> Float[Tensor, 'n_features_1 n_features_2']:
        return self.sums / self.counts


class BackwardImplicationAggregator:
    def __init__(self, layer: int, n_features: Tuple[int, int], lower_bound=0.0):
        """Calculates the pair-wise backward implication of two tensors that are
        provided batch-wise. All computations are done element-wise with einsum.

        Args:
            layer (int): First of the two subsequent layers for similarity computation.
            n_features (Tuple[int, int]): Number of features to include per layer.
            lower_bound (float): Threshold to distinguish between active and inactive.
        """
        self.layer = layer
        self.n_features = n_features
        self.lower_bound = lower_bound

        n_features_1, n_features_2 = n_features

        self.counts = torch.zeros(n_features_1, n_features_2)
        self.sums = torch.zeros(n_features_1, n_features_2)

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        n_features_1, n_features_2 = self.n_features
        
        activations_1 = activations[self.layer, :n_features_1, :]
        activations_2 = activations[self.layer + 1, :n_features_2, :]

        active_1 = (activations_1 > self.lower_bound).float()
        active_2 = (activations_2 > self.lower_bound).float()

        self.counts += active_2.sum(dim=-1).unsqueeze(0)
        self.sums += einops.einsum(active_1, active_2, 'n_features_1 n_tokens, n_features_2 n_tokens -> n_features_1 n_features_2')

    def finalize(self) -> Float[Tensor, 'n_features_1 n_features_2']:
        return self.sums / self.counts
    

class JaccardSimilarityAggregator:
    def __init__(self, layer: int, n_features: Tuple[int, int], lower_bound=0.0):
        """Calculates the pair-wise Jaccard similarity of two tensors that are
        provided batch-wise. All computations are done element-wise with einsum.

        Args:
            layer (int): First of the two subsequent layers for similarity computation.
            n_features (Tuple[int, int]): Number of features to include per layer.
            lower_bound (float): Threshold to distinguish between active and inactive.
        """
        self.layer = layer
        self.n_features = n_features
        self.lower_bound = lower_bound

        n_features_1, n_features_2 = n_features

        self.counts = torch.zeros(n_features_1, n_features_2)
        self.sums = torch.zeros(n_features_1, n_features_2)

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        n_features_1, n_features_2 = self.n_features
        
        activations_1 = activations[self.layer, :n_features_1, :]
        activations_2 = activations[self.layer + 1, :n_features_2, :]

        active_1 = (activations_1 > self.lower_bound).float()
        active_2 = (activations_2 > self.lower_bound).float()
        active_1_2 = einops.einsum(active_1, active_2, 'n_features_1 n_tokens, n_features_2 n_tokens -> n_features_1 n_features_2')

        self.counts += active_1.sum(dim=-1).unsqueeze(1) + active_2.sum(dim=-1).unsqueeze(0) - active_1_2
        self.sums += active_1_2

    def finalize(self) -> Float[Tensor, 'n_features_1 n_features_2']:
        return self.sums / self.counts


class MutualInformationAggregator:
    def __init__(self, layer: int, n_features: Tuple[int, int], lower_bound=0.0):
        """Calculates the pair-wise (binary) mutual information of two tensors that are
        provided batch-wise. All computations are done element-wise with einsum.

        Args:
            layer (int): First of the two subsequent layers for similarity computation.
            n_features (Tuple[int, int]): Number of features to include per layer.
            lower_bound (float): Threshold to distinguish between active and inactive.
        """
        self.layer = layer
        self.n_features = n_features
        self.lower_bound = lower_bound

        n_features_1, n_features_2 = n_features

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        pass

    def finalize(self) -> Float[Tensor, 'n_features_1 n_features_2']:
        
        # Calculate mutual information
        mutual_information = torch.zeros(self.n_features)
        for pxy, px, py in zip([p00, p01, p10, p11], [px0, px0, px1, px1], [py0, py1, py0, py1]):
            mutual_information += (pxy * torch.log(pxy / (px * py))).nan_to_num()

        return mutual_information


class DeadFeaturePairsAggregator:
    def __init__(self, layer: int, n_features: Tuple[int, int], lower_bound: float = 0.0):
        self.layer = layer
        self.n_features = n_features
        self.lower_bound = lower_bound

        self.sums = torch.zeros(n_features, dtype=torch.int)

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        n_features_1, n_features_2 = self.n_features
        
        activations_1 = activations[self.layer, :n_features_1, :]
        activations_2 = activations[self.layer + 1, :n_features_2, :]

        active_1 = (activations_1 > self.lower_bound).float()
        active_2 = (activations_2 > self.lower_bound).float()

        self.sums += einops.einsum(active_1, active_2, 'f1 t, f2 t -> f1 f2').int()

    def finalize(self) -> Int[Tensor, 'n_features_1 n_features_2']:
        return self.sums
    

class DeadFeaturesAggregator:
    def __init__(self, layer: int, n_features: Tuple[int, int], lower_bound: float = 0.0):
        self.layer = layer
        self.n_features = n_features
        self.lower_bound = lower_bound

        self.sums = torch.zeros(n_features, dtype=torch.int)

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        n_features_1, n_features_2 = self.n_features
        
        activations_1 = activations[self.layer, :n_features_1, :]
        activations_2 = activations[self.layer + 1, :n_features_2, :]

        active_1 = (activations_1 > self.lower_bound).float()
        active_2 = (activations_2 > self.lower_bound).float()

        self.sums += einops.einsum(active_1, active_2, 'f1 t, f2 t -> f1 f2').int()

    def finalize(self) -> Int[Tensor, 'n_features_1 n_features_2']:
        return self.sums