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
import os
import shutil
import numpy as np


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
batch_size = 32
batches_per_file = 4
d_sae = 128

folder = 'save_batched_tensors-0'

# Remove the output folder if it exists, and create an empty folder
if os.path.exists(folder):
    shutil.rmtree(folder)  # Remove the directory

os.makedirs(folder)

# For each batch, get activations and append them to the respective files
for batch in range(number_of_batches):
    # If the last file is full, create a new one
    # if batch % batches_per_file == 0:
    #     for layer in range(number_of_layers):
    #         with open(f'{folder}/{layer}_{batch // batches_per_file}.npy', 'wb') as f:
    #             pass

    for layer in range(number_of_layers):
        with open(f'{folder}/{layer}_{batch // batches_per_file}.npy', 'ab') as f:
            layer_activations = np.ones((d_sae, batch_size)) * batch + layer
            np.save(f, layer_activations)


# %%
# Sanity check: Load and concatenate
activations = np.empty((number_of_layers, d_sae, batch_size * number_of_batches))

for layer in range(number_of_layers):
    for file in range(number_of_batches // batches_per_file):
        with open(f'{folder}/{layer}_{file}.npy', 'rb') as f:
            for batch in range(batches_per_file):
                activations[layer, :, ((file * batches_per_file + batch) * batch_size):((file * batches_per_file + batch + 1) * batch_size)] = np.load(f)


# %%
activations[1, :10, :100]
