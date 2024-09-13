"""
as per discussion on 2024/09/09, I will compute
activations and projected errors for a bunch of tokens and features and record them
and collect them and then show them 
"""

# %%
import re
from sae_lens import SAE
from tqdm import tqdm
import torch
import torch as t
import typing
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as t
import torch
import numpy as np
import einops
import typing
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from src import D_SAE
# %%
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"
num_layers = 12
d_model = 768
prepend_bos = True
num_toks_per_row = 128
num_rows = 512
batch_size = 32
MAX_NUM_ACTIVATIONS_PER_LAYER = 500
NECESSITY_DISAPPEAR_THRESHOLD = 0.4
PERCENT_OF_MAX_ACTIVATION = 0.1
# %%
def create_id_to_sae() -> typing.Dict[str, SAE]:
    print("Loading SAEs")
    sae_id_to_sae = {}
    for layer in tqdm(list(range(num_layers))):
        sae_id = f"blocks.{layer}.hook_resid_pre"
        sae, _, _ = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=sae_id,
            device=device
        )
        sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
        sae_id_to_sae[sae_id] = sae
    return sae_id_to_sae
id_to_sae = create_id_to_sae()
# %%
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %%
def load_data() -> DataLoader:
    dataset = load_dataset(path="NeelNanda/pile-10k", split="train", streaming=False)
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=num_toks_per_row,
        add_bos_token=prepend_bos,
    )
    tokens = token_dataset['tokens'][:num_rows]
    print(f"Using {num_rows * num_toks_per_row} tokens")
    return DataLoader(tokens, batch_size=batch_size, shuffle=False)
data_loader = load_data()
# %%
necessity_scores = np.load("artefacts/similarity_measures/necessity_relative_activation/res_jb_sae_feature_similarity_necessity_relative_activation_10M_0.2_0.1.npz")["arr_0"]
# %%
disappearing_features = t.BoolTensor(np.max(necessity_scores, axis=2) <= NECESSITY_DISAPPEAR_THRESHOLD).to(device)
# %%
max_activation_based_threshold = PERCENT_OF_MAX_ACTIVATION * t.Tensor(np.load("artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz")['arr_0']).to(device)

# %%
mask = t.zeros(batch_size, num_toks_per_row, D_SAE).to(device)

sae_activations_per_layer = [
    [] for _ in range(num_layers)
]
error_projections_per_layer = [
    [] for _ in range(num_layers)
]

# %%
PATTERN = re.compile(r'^blocks\.(\d+)\.hook_resid_pre$')
def get_resid_layer_idx(input_string):
    match = PATTERN.match(input_string)
    if match:
        # Extract the number and check if it's greater than 0
        return int(match.group(1))
    return None

def set_acts_and_mask(activations: t.Tensor, hook: HookPoint):
    """
    for a 'prev' layer, define the mask by which features were active and disappearing
    """
    global mask
    sae: SAE = id_to_sae[hook.name]
    sae_acts = sae.encode(activations)
    sae_act_thresholds = max_activation_based_threshold[hook.layer()]
    # which ones DID activate at least 10% of their max?
    max_sae_act_mask = sae_acts > sae_act_thresholds
    mask = max_sae_act_mask & disappearing_features[hook.layer()]
    flattened_selected_acts = sae_acts[mask].reshape(-1)
    sae_activations_per_layer[hook.layer()].extend(flattened_selected_acts.tolist())
    return activations

def set_err_projections(activations: t.Tensor, hook: HookPoint):
    """projects the current layer's error onto the previous layer's feature directions"""
    global mask
    curr_layer_sae: SAE = id_to_sae[hook.name]
    prev_layer_name = f"blocks.{hook.layer()-1}.hook_resid_pre"
    prev_layer_sae: SAE = id_to_sae[prev_layer_name]
    
    curr_layer_sae_error = activations - curr_layer_sae(activations)
    # for each prev layer feature, what was the magnitude of the projection
    dot_product = einops.einsum(
        curr_layer_sae_error, prev_layer_sae.W_dec,
        "batch seq resid, sae resid -> batch seq sae",
    )
    decoder_norms = t.norm(prev_layer_sae.W_dec, dim=1) # should be 1 but just being explicit for readers
    projections = dot_product / decoder_norms
    
    flattened_selected_projections = projections[mask].reshape(-1)
    error_projections_per_layer[hook.layer()-1].extend(flattened_selected_projections.tolist())
    return activations

def is_resid_layer_after_0(input_string):
    curr_layer_idx = get_resid_layer_idx(input_string)
    if curr_layer_idx == None:
        return False
    return curr_layer_idx > 0

def is_resid_layer_before_11(input_string):
    curr_layer_idx = get_resid_layer_idx(input_string)
    if curr_layer_idx == None:
        return False
    return curr_layer_idx < 11
    
# %%
with torch.no_grad():
    for batch_tokens in tqdm(data_loader):
        model.reset_hooks()
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                # compute projections for all layers after 0 (the mask was set for the previous layers)
                (
                    is_resid_layer_after_0,
                    set_err_projections,
                ),
                # set the mask for all layers, in preparation for the next layer to use it
                (
                    is_resid_layer_before_11,
                    set_acts_and_mask,
                ),
            ]
        )
# %%
import matplotlib.pyplot as plt
import numpy as np

def get_limits_for_layer(layer_idx):
    if layer_idx <= 6:
        return -0.5, 15, -15, 15
    elif layer_idx == 7:
        return -0.5, 19, -10, 15
    elif layer_idx == 8:
        return -0.5, 20, -12, 22
    else:
        return -0.5, 30, -15, 38

# %%
import matplotlib as mpl
mpl.rcParams.update({'font.size': mpl.rcParams['font.size'] + 4})
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
def create_heatmaps(sae_activations_per_layer, error_projections_per_layer, layer_idxes):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.05]})
    # Create a list to store all heatmaps for normalization
    all_heatmaps = []

    for i, layer_idx in enumerate(layer_idxes):
        activations = sae_activations_per_layer[layer_idx]
        error_projections = error_projections_per_layer[layer_idx]
        
        # Calculate x-axis and y-axis ranges to cover 99% of the points
        x_min, x_max = 0.3, 4.5
        y_min, y_max = np.percentile(error_projections, [0.5, 99])
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(activations, error_projections, bins=100, 
                                                 range=[[x_min, x_max], [y_min, y_max]])
        all_heatmaps.append(heatmap)

        axes[i].set_title(f'Layer {layer_idx}')
        axes[i].set_xlabel('')  # Remove individual x-labels
        if i == 0:
            axes[i].set_ylabel('Error Projection')
        else:
            axes[i].set_ylabel('')  # Remove y-label for other subplots
        
        # Set x-axis ticks
        x_ticks = np.linspace(x_min, x_max, 9)
        axes[i].set_xticks(np.linspace(0, 100, 9))
        axes[i].set_xticklabels([f'{x:.2f}' for x in x_ticks], rotation=45)

        # Set y-axis ticks
        y_ticks = np.linspace(y_max, y_min, 9)  # Reversed order
        axes[i].set_yticks(np.linspace(0, 100, 9))
        axes[i].set_yticklabels([f'{y:.2f}' for y in y_ticks])

        # Invert y-axis
        axes[i].invert_yaxis()

    # Normalize colormap across all heatmaps
    vmin = min(map(np.min, all_heatmaps))
    vmax = max(map(np.max, all_heatmaps))

    # Plot heatmaps with normalized colormap
    for i, heatmap in enumerate(all_heatmaps):
        im = axes[i].imshow(heatmap.T, cmap='YlOrRd', aspect='auto', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # Create colorbar
    cbar = fig.colorbar(im, cax=axes[-1])
    cbar.set_label('Density', rotation=270, labelpad=15)

    # Add a common x-label
    fig.text(0.5, 0.01, 'Activation Magnitude', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# Example usage:
layer_idxes = [4, 6, 8, 10]
create_heatmaps(sae_activations_per_layer, error_projections_per_layer, layer_idxes)

# %%
