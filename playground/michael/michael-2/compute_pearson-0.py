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
tokens = token_dataset['tokens'][:1024]  # 128k tokens
# tokens = token_dataset['tokens'][:8192]  # 1M tokens


# %%
# Define object for batched co-occurrence computation
class BatchedPearson:
    def __init__(self, shape, lower_bound=0.0, masked=True, device=device):
        """Calculates the pair-wise co-occurrence of two 2d tensors that are provided
        batch-wise.

        Args:
            shape (Size): Shape of the result.
            lower_bound (float, optional): Lower bound for activation. Defaults to 0.0.
            masked (bool, optional): If true, only consider elements where at least one
            of the two tensors is active. Defaults to True.
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


# Define layers and number of features for co-occurrence calculation
layer_1, number_of_features_1 = 6, 10
layer_2, number_of_features_2 = 7, 24576


aggregator = BatchedCooccurrence((number_of_features_1, number_of_features_2))
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.run_with_hooks(batch_tokens)

        # Now we can use sae_activations
        aggregator.process(
            sae_activations[layer_1, :number_of_features_1],
            sae_activations[layer_2, :number_of_features_2]
        )

    cooccurrences = aggregator.finalize()


# %%
px.histogram(cooccurrences[0].cpu())


# %%
(cooccurrences > 0.01).count_nonzero(dim=1)


# %%
cooccurrences[0][cooccurrences[0] > 0.01]
