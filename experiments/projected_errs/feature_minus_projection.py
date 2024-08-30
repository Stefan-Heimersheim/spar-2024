# %%
import numpy as np 
import matplotlib.pyplot as plt
import torch as t
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from sae_lens import SAE
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
# %%
num_layers = 12
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"
# %%
pearson_corrs = np.load("artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_100M_0.1.npz")["arr_0"]
# %%
disappeared_feature_mask = np.max(pearson_corrs, axis=2) <= 0.3

# %%
projected_data = np.load("artefacts/projected_errs/total_num_toks__16777216.npz")
projection_sums = projected_data["projection_sums"]
projection_counts = projected_data['counts'] 
avg_err_projection_onto_feat = projection_sums / projection_counts
# %%
masked_feats_data = np.load("artefacts/projected_errs/avg_masked_feats__total_num_toks__16777216.npz")
activation_sums = masked_feats_data["activation_sums"]
activation_counts = masked_feats_data["counts"]
avg_feat = masked_feats_data["ratios"]
# %%
avg_feat_minus_projection = avg_feat[:-1, :] - avg_err_projection_onto_feat
# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have your matrix avg_feat_minus_projection
# For this example, we'll create random data
avg_feat_minus_projection = np.random.randn(11, 24000)

# Set up the plot
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Histograms of Average Feature Minus Error Projection", fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

for idx, row in enumerate(avg_feat_minus_projection):
    if idx < 11:  # We only have 11 rows, so we'll leave the last plot empty
        ax = axes[idx]
        
        # Calculate histogram
        hist, bin_edges = np.histogram(row, bins=np.arange(min(row), max(row) + 0.2, 0.2))
        
        # Calculate x-limits to include 95% of the data
        sorted_row = np.sort(row)
        lower_bound = sorted_row[int(0.025 * len(row))]
        upper_bound = sorted_row[int(0.975 * len(row))]
        
        # Plot histogram
        ax.hist(row, bins=bin_edges, edgecolor='black')
        
        # Set x-axis limits to cover 95% of the data
        ax.set_xlim(lower_bound, upper_bound)
        
        # Set labels and title
        ax.set_title(f"Layer {idx}")
        ax.set_xlabel("Average Feature Minus Error Projection")
        ax.set_ylabel("Frequency")

# Remove the empty subplot
fig.delaxes(axes[-1])

# Adjust layout and show plot
plt.tight_layout()
plt.show()