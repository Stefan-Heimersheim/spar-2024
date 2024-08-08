# Get SAE activations via hooks to save time and memory
#
# Define a hook that, for each layer, stores model activations of the current batch in
# a global store. SAE activations are then calculated from the model activations.

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

tokens = token_dataset['tokens']

# %%
len(token_dataset)


# %%
# OPTIONAL: Reduce dataset for faster experimentation
tokens = tokens[:1024]


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

with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.run_with_hooks(batch_tokens)

        # Now we can use sae_activations
