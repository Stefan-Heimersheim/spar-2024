# %%
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sae_lens import SAE
from tqdm import tqdm
import torch
import torch as t
import typing
import numpy as np
from scipy import integrate
from datasets import load_dataset
from torch.utils.data import DataLoader
import einops
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
disappeared_feature_mask = np.max(pearson_corrs, axis=2) <= 0.4

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
import numpy as np
import matplotlib.pyplot as plt

def create_histogram_overlay_plots(avg_err_projection_onto_feat, avg_feat, disappeared_feature_mask):
    assert avg_err_projection_onto_feat.shape == avg_feat.shape == disappeared_feature_mask.shape
    num_plots, _ = avg_err_projection_onto_feat.shape

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))  # 4x3 grid for 11 plots (12th will be empty)
    axes = axes.flatten()

    # Create empty lists to store plot objects for the legend
    blue_plots = []
    red_plots = []

    for idx in range(num_plots):
        ax = axes[idx]
        
        # Get the mask for this index
        mask = disappeared_feature_mask[idx, :]
        
        # Filter the data based on the mask and remove NaN values
        err_data = avg_err_projection_onto_feat[idx, mask]
        feat_data = avg_feat[idx, mask]
        
        # Remove NaN values
        err_data = err_data[~np.isnan(err_data)]
        feat_data = feat_data[~np.isnan(feat_data)]
        
        if len(err_data) == 0 and len(feat_data) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            ax.set_title(f'Layer {idx}')
            continue
        
        # Combine all data for percentile calculation
        all_data = np.concatenate([err_data, feat_data])
        
        # Calculate 97.5th percentile for upper x-axis limit
        upper_limit = np.percentile(all_data, 97.5)
        
        # Set lower limit to -0.2
        lower_limit = -0.2
        
        # Determine the range for bins
        bins = np.arange(lower_limit, upper_limit + 0.2, 0.2)
        
        # Plot histogram for avg_err_projection_onto_feat
        blue_plot = ax.hist(err_data, bins=bins, alpha=0.5, color='blue', label='Avg Next Layer Error Projection')
        
        # Plot histogram for avg_feat
        red_plot = ax.hist(feat_data, bins=bins, alpha=0.5, color='red', label='Average Prev Layer Feature')
        
        # Store plot objects for legend
        blue_plots.append(blue_plot)
        red_plots.append(red_plot)
        
        # Set x-axis limits
        ax.set_xlim(lower_limit, upper_limit)
        
        # Update the title to show "Layer {idx}"
        ax.set_title(f'Layer {idx}')
        
        # Set x-label to "Magnitude" and y-label
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')

    # Remove the empty 12th subplot
    fig.delaxes(axes[-1])

    # Create a single legend for the entire figure
    fig.legend([blue_plots[0][2][0], red_plots[0][2][0]], 
               ['Avg Next Layer Error Projection', 'Average Prev Layer Feature'],
               loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    plt.tight_layout()
    # Adjust the bottom margin to make room for the legend
    plt.subplots_adjust(bottom=0.1)
    plt.show()

# Example usage:
# avg_err_projection_onto_feat = np.random.randn(11, 2000)
# avg_feat = np.random.randn(11, 2000)
# disappeared_feature_mask = np.random.choice([True, False], size=(11, 2000))
# # Introduce some NaN values and outliers
# avg_err_projection_onto_feat[np.random.rand(*avg_err_projection_onto_feat.shape) < 0.1] = np.nan
# avg_feat[np.random.rand(*avg_feat.shape) < 0.1] = np.nan
# avg_err_projection_onto_feat[np.random.rand(*avg_err_projection_onto_feat.shape) < 0.01] = 1000
# avg_feat[np.random.rand(*avg_feat.shape) < 0.01] = -1000
# create_histogram_overlay_plots(avg_err_projection_onto_feat, avg_feat, disappeared_feature_mask)
# %%
import numpy as np
import matplotlib.pyplot as plt

def create_scatterplots(avg_err_projection_onto_feat, avg_feat, disappeared_feature_mask):
    assert avg_err_projection_onto_feat.shape == avg_feat.shape == disappeared_feature_mask.shape
    num_plots, _ = avg_err_projection_onto_feat.shape

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 4x3 grid for 11 plots (12th will be empty)
    fig.suptitle("Avg Feature Value vs Next Layer Error Projection for active features which disappeared", fontsize=16)
    axes = axes.flatten()

    for idx in range(num_plots):
        ax = axes[idx]
        
        # Get the mask for this index
        mask = disappeared_feature_mask[idx, :]
        
        # Filter the data based on the mask and remove NaN values
        err_data = avg_err_projection_onto_feat[idx, mask]
        feat_data = avg_feat[idx, mask]
        
        # Remove entries where either err_data or feat_data is NaN, or feat_data < 0.01
        valid_mask = ~(np.isnan(err_data) | np.isnan(feat_data)) & (feat_data >= 0.01)
        err_data = err_data[valid_mask]
        feat_data = feat_data[valid_mask]
        
        if len(err_data) == 0 and len(feat_data) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            ax.set_title(f'Layer {idx}')
            continue
        
        # Create scatterplot
        ax.scatter(feat_data, err_data, alpha=0.5, s=1)  # s=1 for small point size
        
        # Set x-axis limits
        x_min, x_max = 0.01, np.percentile(feat_data, 97.5)
        ax.set_xlim(x_min, x_max)
        
        # Set y-axis limits
        y_min = min(0, np.percentile(err_data, 2.5))
        y_max = np.percentile(err_data, 97.5)
        ax.set_ylim(y_min, y_max)
        
        # Update the title to show "Layer {idx}"
        ax.set_title(f'Layer {idx} Features vs Layer {idx+1} Error')
        
        # Set x-label and y-label
        ax.set_xlabel(f'Layer {idx} Feature Magnitude')
        ax.set_ylabel(f'Layer {idx+1} Error Projected Onto L{idx} SAE W_dec')

        # Add a red line for y=x
        line, = ax.plot([x_min, max(x_max, y_max)], 
                        [x_min, max(x_max, y_max)], 
                        color='r', linestyle='--', linewidth=0.5)

    # Use the 12th subplot for the legend
    legend_ax = axes[-1]
    legend_ax.axis('off')
    legend_ax.legend([line], ['y=x'], loc='center')

    plt.tight_layout()
    # Adjust the layout to make room for the overall title
    plt.subplots_adjust(top=0.95)
    plt.show()

# Example usage:
# avg_err_projection_onto_feat = np.random.randn(11, 2000)
# avg_feat = np.abs(np.random.randn(11, 2000))  # Using abs to ensure positive values
# disappeared_feature_mask = np.random.choice([True, False], size=(11, 2000))
# # Introduce some NaN values and outliers
# avg_err_projection_onto_feat[np.random.rand(*avg_err_projection_onto_feat.shape) < 0.1] = np.nan
# avg_feat[np.random.rand(*avg_feat.shape) < 0.1] = np.nan
# avg_err_projection_onto_feat[np.random.rand(*avg_err_projection_onto_feat.shape) < 0.01] = 1000
# avg_feat[np.random.rand(*avg_feat.shape) < 0.01] = 1000
# create_scatterplots(avg_err_projection_onto_feat, avg_feat, disappeared_feature_mask)
# %%
