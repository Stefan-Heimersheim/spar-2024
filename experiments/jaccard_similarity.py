# %%
import argparse
from datetime import datetime
from functools import partial
import numpy as np
import torch as t
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import einops
from torch.utils.data import DataLoader
from tqdm import tqdm
import plotly.express as px
from src.similarity_helpers import save_compressed


# %%
# Config
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
# Load Model, SAEs and data
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

print(f'Loading SAEs')
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
# OPTIONAL: Reduce dataset for faster experimentation
tokens = token_dataset['tokens'][:1024]


# %%
# Define object for batched Jaccard similarity computation
class BatchedJaccard:
    def __init__(self, shape, lower_bound=0.0, masked=True, device=device, epsilon=1e-6):
        """Calculates the pair-wise Jaccard similarity of two 2d tensors that are provided
        batch-wise.

        Args:
            shape (Size): Shape of the result.
            lower_bound (float, optional): Lower bound for activation. Defaults to 0.0.
            masked (bool, optional): If true, only consider elements where at least one
            of the two tensors is active. Defaults to True.
        """
        self.count = epsilon * t.ones(shape).to(device) if masked else 0
        self.sums = t.zeros(shape).to(device)

        self.lower_bound = lower_bound
        self.masked = masked

    def process(self, tensor_1, tensor_2):
        active_1 = tensor_1 > self.lower_bound
        active_2 = tensor_2 > self.lower_bound


        if self.masked:
            active_1_unsq = active_1.unsqueeze(1)  # Shape: (n_feats1, 1, seq_len)
            active_2_unsq = active_2.unsqueeze(0)  # Shape: (1, n_feats2, seq_len)
            or_result = (active_1_unsq | active_2_unsq).sum(dim=2)  # Shape: (n_feats1, n_feats2)
            and_result = (active_1_unsq & active_2_unsq).sum(dim=2)  # Shape: (n_feats1, n_feats2)
            self.count += or_result
            self.sums += and_result

        else:
            self.count += tensor_1.shape[-1]
            same_result = (active_1.unsqueeze(1) == active_2.unsqueeze(0)).sum(dim=2)  # Shape: (n_feats1, n_feats2)
            self.sums += same_result


    def finalize(self):
        return (self.sums / self.count)

# %%
batch_size = 32  # Batch size of 32 seems to be optimal for model run-time
data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)

# %%
def get_layer_jaccard(layer, layer_similarities, batch_size=32, feat_batch_size=64, data_loader=data_loader):
    print(f'Computing Jaccard similarity for layer {layer}')
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    saes = []
    for i in range(2):
        sae, _, _ = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer+i}.hook_resid_pre",
            device=device
        )
        saes.append(sae)
    d_sae = saes[0].cfg.d_sae
    context_size = saes[0].cfg.context_size
    sae_activations = t.empty(2, d_sae, batch_size * context_size).to(device)
    print(f'd_sae: {d_sae}, context_size: {context_size}, batch_size: {batch_size}')
    print(f'sae_activations shape: {sae_activations.shape}')

    def retrieval_hook(activations, hook, idx=0):
        layer = hook.layer()
        sae_activations[idx] = einops.rearrange(
            saes[idx].encode(activations),
            'batch seq features -> features (batch seq)'
        )
    model.add_hook(f'blocks.{layer}.hook_resid_pre', partial(retrieval_hook, idx=0))
    model.add_hook(f'blocks.{layer+1}.hook_resid_pre', partial(retrieval_hook, idx=1))

    n_feats_1 = feat_batch_size
    n_feats_2 = d_sae
    n_feat_batches = d_sae // feat_batch_size
    aggregators = {i: BatchedJaccard((n_feats_1, n_feats_2)) for i in range(n_feat_batches)}
    with t.no_grad():
        for batch_tokens in tqdm(data_loader):
            model.run_with_hooks(batch_tokens)
            for i_feat in tqdm(range(n_feat_batches)):
                feat_start = i_feat * n_feats_1
                feat_end = (i_feat + 1) * n_feats_1
                aggregator = aggregators[i_feat]
                aggregator.process(
                    sae_activations[0, feat_start:feat_end],
                    sae_activations[1, :n_feats_2]
                )
        for i_feat in range(n_feat_batches):
            aggregator = aggregators[i_feat]
            similarity = aggregator.finalize()
            layer_similarities[feat_start:feat_end, :] = similarity

# %%
# layer_similarities = t.empty(d_sae, d_sae).cpu()
# layer_0_similarities = get_layer_jaccard(layer=0, layer_similarities=layer_similarities, batch_size=32, feat_batch_size=64, data_loader=data_loader)
# %%
def get_all_layer_similarities(layers, filename=None, batch_size=32, feat_batch_size=64, data_loader=data_loader):
    for layer in tqdm(layers):
        layer_similarities = t.empty(d_sae, d_sae).cpu()
        get_layer_jaccard(layer=layer, layer_similarities=layer_similarities, batch_size=batch_size, feat_batch_size=feat_batch_size, data_loader=data_loader)
        if filename is not None:
            save_filename = f'{filename}_layer_{layer}.npz'
            print(f'Saving Jaccard similarity for layer {layer} to {save_filename}')
            save_compressed(layer_similarities.detach().cpu().numpy(), save_filename)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='All layers')
    parser.add_argument('-l', type=int, help='Specific layer', default=None)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument('-f', type=str, help='Filename', default=f'jaccard_similarity_{timestamp}')
    parser.add_argument('-fb', type=int, help='Feature batch size', default=64)
    args = parser.parse_args()
    if args.a:
        print(f'Computing Jaccard similarity for all layers')
        get_all_layer_similarities(layers=list(range(12)), filename=args.f, batch_size=32, feat_batch_size=args.fb, data_loader=data_loader)
    elif args.l is not None:
        layer_similarities = t.empty(d_sae, d_sae).cpu()
        get_layer_jaccard(layer=args.l, layer_similarities=layer_similarities, batch_size=32, feat_batch_size=args.fb, data_loader=data_loader)
        filename = f'{args.f}_layer_{args.l}.npz'
        print(f'Saving Jaccard similarity for layer {args.l} to {filename}')
        save_compressed(layer_similarities.detach().cpu().numpy(), filename)