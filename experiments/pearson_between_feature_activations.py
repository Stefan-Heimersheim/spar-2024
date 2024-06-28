# Get SAE activations via hooks to save time and memory
#
# Define a hook that, for each layer, stores model activations of the current batch in
# a global store. SAE activations are then calculated from the model activations.
# From the SAE activations, we calculate the Person correlation between features.

# %%
# Imports
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import einops
from torch.utils.data import DataLoader
from tqdm import tqdm
import plotly.express as px


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

saes = []
for layer in tqdm(list(range(model.cfg.n_layers))):
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=device
    )
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    saes.append(sae)

# These hyperparameters are used to pre-process the data
context_size = saes[0].cfg.context_size
prepend_bos = saes[0].cfg.prepend_bos
d_sae = saes[0].cfg.d_sae

dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=context_size,
    add_bos_token=prepend_bos,
)

tokens = token_dataset['tokens']  # 17.5M tokens


# %%
# OPTIONAL: Reduce dataset for faster experimentation
# tokens = token_dataset['tokens'][:32]  # 4k tokens
tokens = token_dataset['tokens'][:1024]  # 128k tokens
# tokens = token_dataset['tokens'][:8192]  # 1M tokens


# %%
# Define object for batched co-occurrence computation
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

        self.nonzero_counts_1 = torch.zeros(shape[0])
        self.nonzero_counts_2 = torch.zeros(shape[1])

    def process(self, tensor_1, tensor_2):
        self.count += tensor_1.shape[-1]

        self.sums_1 += tensor_1.sum(dim=-1)
        self.sums_2 += tensor_2.sum(dim=-1)

        self.sums_of_squares_1 += (tensor_1 ** 2).sum(dim=-1)
        self.sums_of_squares_2 += (tensor_2 ** 2).sum(dim=-1)

        self.sums_1_2 += einops.einsum(tensor_1, tensor_2, 'f1 t, f2 t -> f1 f2')

        self.nonzero_counts_1 += tensor_1.count_nonzero(dim=-1)
        self.nonzero_counts_2 += tensor_2.count_nonzero(dim=-1)

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


# %%
batch_size = 32  # Batch size of 32 seems to be optimal for model run-time
data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

sae_activations = torch.empty(model.cfg.n_layers, d_sae, batch_size * context_size)


def retrieval_hook(activations, hook):
    layer = hook.layer()

    sae_activations[layer] = einops.rearrange(
        saes[layer].encode(activations),
        'batch seq features -> features (batch seq)'
    )


model.add_hook(lambda name: name.endswith('.hook_resid_pre'), retrieval_hook)


# Define layers and features for co-occurrence calculation
layer_1, number_of_features_1 = 6, 24576
layer_2, number_of_features_2 = 7, 24576


aggregator = BatchedPearson((number_of_features_1, number_of_features_2))
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.run_with_hooks(batch_tokens)

        # Now we can use sae_activations
        aggregator.process(
            sae_activations[layer_1, :number_of_features_1],
            sae_activations[layer_2, :number_of_features_2]
        )

    pearson_correlations = aggregator.finalize()

# `pearson_correlations` is now a (number_of_features_1, number_of_features_2) tensor
# that stores the correlation values for each pair of features


# %%
px.histogram(pearson_correlations[0][pearson_correlations[0] > 0].cpu(), title='Histogram of Pearson correlations from 6/0 to 7/:')
