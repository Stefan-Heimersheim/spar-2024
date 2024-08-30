import torch
from jaxtyping import Float
from torch import Tensor

from similarity_measures import Aggregator


class MaxActivationAggregator(Aggregator):
    def __init__(self, n_layers: int, d_sae: int, device='cuda', **kwargs):
        """Collects the maximum activation for all layers and
        SAE features.

        Args:
            n_layers (int): The number of model layers.
            d_sae (int): The number of features in each SAE.
        """
        self.max_activations = torch.zeros(n_layers, d_sae, device=device)

    def process(self, activations: Float[Tensor, 'n_layers n_features n_tokens']) -> None:
        self.max_activations = self.max_activations.maximum(activations.max(dim=-1)[0])

    def finalize(self) -> Float[Tensor, 'n_features_1 n_features_2']:
        return self.max_activations
