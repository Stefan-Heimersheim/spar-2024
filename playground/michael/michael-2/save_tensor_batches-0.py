# Save activations in files, appending batches as they are processed
#
# This is a test with artificial activation tensors.

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
number_of_layers = 12
number_of_batches = 16

# Create an empty tensor file per layer


# For each batch, get activations and append them to the respective files
for batch in range(number_of_batches):
    