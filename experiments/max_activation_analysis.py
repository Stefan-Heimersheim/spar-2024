# Compute max SAE features activations for all SAEs and store them as a (n_layers, d_sae) = (12, 24576) numpy array.

# %%
# Imports
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline_helpers import load_model_and_saes, load_data, run_with_aggregator
from utils import MaxActivationAggregator


# %%
# Config
device = 'cuda:1'
number_of_batches, number_of_token_desc = 4269, '17.5M'
artefacts_folder = '../artefacts'


# %%
# Load model, SAEs, and data
print('Loading model and SAEs...')
model, saes = load_model_and_saes(model_name='gpt2-small', sae_name='gpt2-small-res-jb', hook_name='hook_resid_pre', device=device)

tokens = load_data(model, saes[0], dataset_name='NeelNanda/pile-10k', number_of_batches=number_of_batches).to(device)


# %%
# Run model and collect max activations
n_layers = model.cfg.n_layers
d_sae = saes[0].cfg.d_sae

aggregator = MaxActivationAggregator(n_layers, d_sae, device=device)
max_activations = run_with_aggregator(model, saes, 'hook_resid_pre', tokens, aggregator, device)

# Save max activations as numpy array
np.savez_compressed(f'{artefacts_folder}/max_activation_analysis/res_jb_sae_max_activations_{number_of_token_desc}', max_activations.cpu().numpy())


# %%
# [OPTIONAL] Load max activations from file
# max_activations = np.load('{artefacts_folder}/max_activation_analysis/res_jb_sae_max_activations_{number_of_token_desc}.npz')['arr_0']


# %%
# Plot histogram of max activations
data = max_activations.flatten()

fig, ax = plt.subplots(figsize=(10, 6))
min, max = 0, math.ceil(data.max() / 100) * 100
bins = np.linspace(min, max, 1000)

hist, bins, _ = ax.hist(data, bins=bins)

ax.set_yscale('log')
ax.set_xlabel(f'Maximum activation over {number_of_token_desc} tokens (log scale)')
ax.set_ylabel('Number of SAE features across all layers')
ax.set_title('Histogram of maximum activation per SAE feature')
ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()

plt.savefig(f'{artefacts_folder}/max_activation_analysis/res_jb_sae_max_activations_{number_of_token_desc}.png', dpi=300, format='png', bbox_inches='tight')