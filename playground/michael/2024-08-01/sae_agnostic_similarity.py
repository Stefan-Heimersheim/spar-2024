# %%
import os
# OPTIONAL: Set environment variable to control visibility of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import einops
import numpy as np
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
import sae_lens
from abc import ABC
from torch.utils.data import DataLoader
from tqdm import tqdm
import blobfile as bf
import sparse_autoencoder
import hashlib
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from similarity_measures import Aggregator, PearsonCorrelationAggregator


# %%
# OPTIONAL: Check if the correct GPU is visible
print(torch.cuda.device_count())  # Should print 1
print(torch.cuda.current_device())  # Should print 0 since it's the first visible device
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
# Define number of tokens
# number_of_batches, number_of_token_desc = 1, '4k'
# number_of_batches, number_of_token_desc = 32, '128k'
number_of_batches, number_of_token_desc = 256, '1M'
# number_of_batches, number_of_token_desc = 4269, '17.5M'


# %%
# src functions
class SAE(ABC):
    def encode(self, activations):
        pass

    @property
    def d_sae(self, ):
        pass

    @property
    def context_size(self, ):
        pass

    @property
    def prepend_bos(self, ):
        pass


class SAELens_SAE(SAE):
    def __init__(self, sae) -> None:
        super().__init__()

        self.sae = sae

    def encode(self, activations):
        return self.sae.encode(activations)

    @property
    def d_sae(self):
        return self.sae.cfg.d_sae
    
    @property
    def context_size(self):
        return self.sae.cfg.context_size
    
    @property
    def prepend_bos(self):
        return self.sae.cfg.prepend_bos


class OpenAI_SAE(SAE):
    def __init__(self, sae, d_sae, context_size, prepend_bos) -> None:
        super().__init__()

        self.sae = sae

        self._d_sae = d_sae
        self._context_size = context_size
        self._prepend_bos = prepend_bos

    def encode(self, activations):
        sae_activations, info = self.sae.encode(activations)

        return sae_activations
    
    @property
    def d_sae(self):
        return self._d_sae
    
    @property
    def context_size(self):
        return self._context_size
    
    @property
    def prepend_bos(self):
        return self._prepend_bos


def load_model(model_name, device):
    return HookedTransformer.from_pretrained(model_name, device=device)


def load_sae_lens_saes(model, sae_name, hook_name, device='cuda'):
    saes = []
    for layer in tqdm(list(range(model.cfg.n_layers))):
        sae, _, _ = sae_lens.SAE.from_pretrained(
            release=sae_name,
            sae_id=f"blocks.{layer}.{hook_name}",
            device=device,
        )
        sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
        saes.append(SAELens_SAE(sae))

    return saes


def download_and_return_openai_sae(url, cache_dir='.cache', verbose=False):
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename based on the URL
    filename = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, filename)
    
    # Check if the file is already cached
    if os.path.exists(cache_path):
        if verbose:
            print(f"File found in cache: {cache_path}")
        
        with open(cache_path, 'rb') as f:
            return torch.load(f)
    
    # If not cached, download the file
    if verbose:
        print(f"Downloading file from {url}...")
    
    # Save the downloaded content to the cache
    with bf.BlobFile(url, 'rb') as f:
        content = f.read()

    with open(cache_path, 'wb') as cf:
        cf.write(content)

    if verbose:
        print(f"File cached at: {cache_path}.")

    with bf.BlobFile(url, 'rb') as f:
        return torch.load(f)


def load_openai_saes(model, hook_name, device='cuda'):
    openai_hook_names = {
        "mlp.hook_post": "mlp_post_act",
        "hook_attn_out": "resid_delta_attn",
        "hook_resid_mid": "resid_post_attn",
        "hook_mlp_out": "resid_delta_mlp",
        "hook_resid_post": "resid_post_mlp",
    }
    
    saes = []
    for layer in tqdm(list(range(model.cfg.n_layers))):
        state_dict = download_and_return_openai_sae(sparse_autoencoder.paths.v5_32k(openai_hook_names[hook_name], layer))
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
        autoencoder.to(device)

        saes.append(OpenAI_SAE(autoencoder, d_sae=32768, context_size=128, prepend_bos=True))

    return saes


def load_data(model, sae, dataset_name, number_of_batches=None, batch_size=32):
    context_size = sae.context_size
    prepend_bos = sae.prepend_bos
    
    dataset = load_dataset(path=dataset_name, split="train", streaming=False)

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=context_size,
        add_bos_token=prepend_bos
    )

    # Cut off to avoid a partial batch at the end
    tokens = token_dataset["tokens"][:((len(token_dataset) // batch_size) * batch_size)]

    if number_of_batches is None:
        return tokens
    else:
        return tokens[:(number_of_batches * batch_size)]


def load_saes(model, sae_name, device):
    assert sae_name in ['sae_lens', 'openai']

    if sae_name == 'sae_lens':
        return load_sae_lens_saes(model, sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)
    else:
        return load_openai_saes(model, hook_name='hook_resid_post', device=device)
    

def run_with_aggregator(model, saes, hook_name, tokens, aggregator : Aggregator, batch_size=32):
    data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

    context_size = saes[0].context_size
    d_sae = saes[0].d_sae
    sae_activations = torch.empty(model.cfg.n_layers, d_sae, batch_size * context_size)


    def retrieval_hook(activations, hook):
        layer = hook.layer()

        sae_activations[layer] = einops.rearrange(
            # Get SAE activations
            saes[layer].encode(activations), "batch seq features -> features (batch seq)"
        )


    model.reset_hooks()
    model.add_hook(lambda name: name.endswith(f".{hook_name}"), retrieval_hook)

    with torch.no_grad():
        for batch_tokens in tqdm(data_loader):
            model.run_with_hooks(batch_tokens)

            # Now we can use sae_activations
            aggregator.process(sae_activations)

        return aggregator.finalize()

# %%
# Load model, SAEs and data
model = load_model(model_name='gpt2-small', device=device)
saes = load_saes(model, sae_name='openai', device=device)
tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=number_of_batches)


# %%
# Run experiment
d_sae = saes[0].d_sae
layer = 0
aggregator = PearsonCorrelationAggregator(layer, (d_sae, d_sae))
result = run_with_aggregator(model, saes, 'hook_resid_pre', tokens, aggregator)

result.shape


# %%
# Compute a similarity measure:
# - Load model, saes and data
# - Create output folder
# - Run model and get SAE activations for each layer individually
# - Store activations in matrix files

measure_name = 'pearson_correlation'
output_folder = f'../../../artefacts/similarity_measures/{measure_name}/.unclamped'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

activity_lower_bound = 0.0
output_filename_fn = lambda layer: f'{output_folder}/openai_sae_feature_similarity_{measure_name}_{number_of_token_desc}_{activity_lower_bound}_{layer}.npz'

d_sae = saes[0].d_sae

for layer in [10]: # range(model.cfg.n_layers - 1):
    aggregator = PearsonCorrelationAggregator(layer, (d_sae, d_sae))

    pearson_correlations = run_with_aggregator(model, saes, 'hook_resid_pre', tokens, aggregator)

    np.savez_compressed(output_filename_fn(layer), pearson_correlations)
