# Analyze the distribution of SAE feature activations
#
# The main goal of this analysis is to determine the most sensible way
# to distinguish active and non-active features on the per-token level.
# 


# %%
# Imports
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import einops
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


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
        device=device,
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

tokens = token_dataset["tokens"][:136_608]  # 17.5M tokens


# %%
# OPTIONAL: Reduce dataset for faster experimentation
# tokens = token_dataset['tokens'][:32]  # 4k tokens
tokens = token_dataset['tokens'][:1024]  # 128k tokens
# tokens = token_dataset["tokens"][:8192]  # 1M tokens


# %%
with open('../../../artefacts/res_jb_max_sae_activations.pt', 'rb') as f:
    max_activations = torch.load(f)


# %%
for layer in range(12):
    plt.hist(max_activations[layer], bins=500)
    plt.show()


# %%
# Number of dead features per layer
(max_activations == 0).sum(dim=-1)


# %%
class BatchedActivationHistogram:
    def __init__(self, n_layers, n_features, max_activations, n_bins):
        # Init bins
        self.bins = torch.empty(n_layers, n_features, n_bins+1)
        for layer in range(n_layers):
            for feature in range(n_features):
                self.bins[layer, feature] = torch.linspace(0, max_activations[layer, feature], steps=n_bins + 1)

        # Init histogram
        self.hist = torch.zeros(n_layers, n_features, n_bins)

    def process(self, activations):
        n_layers, n_features, _ = activations.shape

        for layer in range(n_layers):
            for feature in range(n_features):
                h, _ = torch.histogram(activations[layer, feature], bins=self.bins[layer, feature])

                self.hist[layer, feature] += h

    def finalize(self):
        return self.hist


# %%
batch_size = 32  # Batch size of 32 seems to be optimal for model run-time
data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

sae_activations = torch.empty(model.cfg.n_layers, d_sae, batch_size * context_size)


def retrieval_hook(activations, hook):
    layer = hook.layer()

    sae_activations[layer] = einops.rearrange(
        saes[layer].encode(activations), "batch seq features -> features (batch seq)"
    )


model.add_hook(lambda name: name.endswith(".hook_resid_pre"), retrieval_hook)

n_bins = 100
aggregator = BatchedActivationHistogram(model.cfg.n_layers, d_sae, max_activations, n_bins=n_bins)
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.run_with_hooks(batch_tokens)

        # Now we can use sae_activations
        aggregator.process(sae_activations)

histogram = aggregator.finalize()  # (n_layers, d_sae, n_bins)

np.savez_compressed(f"res_jb_sae_feature_activation_histogram_{n_bins}_bins.npz", histogram)


# %%
plt.hist(histogram[:, :, 1:].sum(dim=-1).flatten(), bins=500)


# %%
plt.hist(histogram[5, 1567, :1], bins=500)