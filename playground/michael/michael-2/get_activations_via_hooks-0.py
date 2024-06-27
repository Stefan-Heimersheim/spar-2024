# Get model activations via hooks to save time and memory
#
# Define a hook that, for each layer, stores model activations of the current batch in
# a global store. For this store, they can be processed further, e.g., by calculating
# SAE activations.

# %%
# Imports
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
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

dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=128,
    add_bos_token=True,
)

tokens = token_dataset['tokens']


# %%
# OPTIONAL: Reduce dataset for faster experimentation
tokens = tokens[:1024]


# %%
batch_size = 32  # Batch size of 32 seems to be optimal for model run-time
data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

model_activations = torch.empty(model.cfg.n_layers, model.cfg.d_model, batch_size * 128)


def retrieval_hook(activations, hook):
    model_activations[hook.layer()] = einops.rearrange(
        activations, 'batch pos d_model -> d_model (batch pos)'
        )


model.add_hook(lambda name: name.endswith('.hook_resid_pre'), retrieval_hook)

with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.run_with_hooks(batch_tokens)


# %%
print(model_activations.shape)
