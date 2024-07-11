# Compute max SAE features activations for all SAEs and store them in
# `res_jb_max_sae_activations.pt` as a (n_layers, d_sae) = (12, 24576) tensor.

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
    device = 'mps'
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device: {device}')


# %%
# Load Model, SAEs and data
model = HookedTransformer.from_pretrained('gpt2-small', device=device)

saes = []
for layer in tqdm(list(range(model.cfg.n_layers))):
    sae, _, _ = SAE.from_pretrained(
        release='gpt2-small-res-jb',
        sae_id=f'blocks.{layer}.hook_resid_pre',
        device=device
    )
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    saes.append(sae)

# These hyperparameters are used to pre-process the data
context_size = saes[0].cfg.context_size
prepend_bos = saes[0].cfg.prepend_bos
d_sae = saes[0].cfg.d_sae

dataset = load_dataset(path='NeelNanda/pile-10k', split='train', streaming=False)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=context_size,
    add_bos_token=prepend_bos,
)

# 17.5M tokens (136,608 rows with 128 tokens each)
# The dataset actually has 136,625 rows, but we want to have a multiple
# of 32 to get evenly sized batches
tokens = token_dataset['tokens'][:136_608]


# %%
class BatchedMaxActivation:
    def __init__(self, n_layers, d_sae):
        """Collects the maximum activation for all layers and
        SAE features.

        Args:
            n_layers (int): The number of model layers.
            d_sae (int): The number of features in each SAE.
        """
        self.max_activations = torch.zeros(n_layers, d_sae)

    def process(self, tensor_1):
        self.max_activations = self.max_activations.maximum(tensor_1.max(dim=-1)[0])

    def finalize(self):
        return self.max_activations


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

aggregator = BatchedMaxActivation(len(saes), d_sae)
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.run_with_hooks(batch_tokens)

        # Now we can use sae_activations
        aggregator.process(sae_activations)

    max_activations = aggregator.finalize()

    with open('res_jb_max_sae_activations.pt', 'wb') as f:
        torch.save(max_activations, f)


# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

x = np.load('../../artefacts/max_sae_activations/res_jb_max_sae_activations.npz')['arr_0']
# %%
plt.hist(x.flatten(), bins=1000)
plt.title('Histogram of maximum activation per feature')
plt.ylabel('Number of SAE features (across layers)')
plt.xlabel('Maximum activation over 17.5M tokens')