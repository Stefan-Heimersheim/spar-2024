# %%
import os
import torch
import einops
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data


# %%
device = 'cuda:1'
n_samples = 100


# %%
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)
tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=1, batch_size=n_samples)
max_activations = torch.tensor(np.load('../../artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz')['arr_0'])


# %%
def get_active_features(model, saes, tokens, lower_bound, relative_bound, hook_name='hook_resid_pre'):
    n_layers = model.cfg.n_layers
    n_features = saes[0].cfg.d_sae
    batch_size, context_size = tokens.shape

    if relative_bound:
        lower_bound = lower_bound.unsqueeze(2).unsqueeze(3)

    sae_activations = torch.empty(n_layers, batch_size, context_size, n_features)

    def retrieval_hook(activations, hook):
        layer = hook.layer()

        sae_activations[layer] = saes[layer].encode(activations)

    model.add_hook(lambda name: name.endswith(f".{hook_name}"), retrieval_hook)

    with torch.no_grad():
        model.run_with_hooks(tokens)

        # Now we can use sae_activations
        return einops.rearrange((einops.rearrange(sae_activations, 'n_layers n_samples n_tokens n_features -> n_layers n_features n_samples n_tokens') > lower_bound)[:, :, :, -1].bool(), 'n_layers n_features n_samples -> n_samples n_layers n_features')


# %%
# Plot number of active features for different lower bounds
lower_bounds = np.linspace(0, 1, 20)
mean_active_features = [get_active_features(model, saes, tokens, lower_bound=lower_bound * max_activations, relative_bound=True).sum(dim=(1, 2)).float().mean() for lower_bound in tqdm(lower_bounds)]

plt.plot(lower_bounds, mean_active_features)
plt.title(f'Mean number of active features for a single token')
plt.xlabel('Relative activation bound')
plt.ylabel('Number of active features (across all layers)')
plt.show()


# %%
# Save matrix for a specific (relative) lower bound
lower_bound = 0.0

any_active = get_active_features(model, saes, tokens, lower_bound=lower_bound * max_activations, relative_bound=True)
print(any_active.sum(dim=(1, 2)).float().mean())


# %%
folder = '../../artefacts/active_features'
if not os.path.exists(folder):
    os.makedirs(folder)

np.savez_compressed(f'{folder}/res_jb_sae_active_features_rel_{lower_bound:.1f}_{n_samples}_last.npz', any_active)
