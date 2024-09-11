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
"""
I have three arrays of size (11, 20000), called avg_err_projection_onto_feat and avg_featwhich have floats and and disappeared_feature_mask which has booleans.

I want to create four scatterplots: for the following layer_idxes [1, 4, 7, 10] I want to plot avg_err_projection_onto_feat[idx, disappeared_feature_mask[idx, :]]​against avg_feat[disappeared_feature_mask[idx, :]]​. I want them all on the same plot. I want each layer's points to colored dark red, light orange, light green, dark blue respectively. I want to draw a line of best fit through each layer's points, matching the color of the points. Then I want to calculate the line of best fit through each of the four layer's points taken together, colored black. I want a legend which indicates which color is which layer, and the black line marked as "layers {1, 4, 7, 10}" in the legend.

Also plot a grey dotted line on y=x and mark that in the legend.

There are nans in the data, make sure to remove them.

Ensure that the x-axis limits are enough to capture 98% of all the x-values across each of the four layers.

The x-axis label should be "Feature Magnitude" and the y-axis label should be "Next Layer Error Projected Onto Prev Layer SAE W_dec"
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming avg_err_projection_onto_feat, avg_feat, and disappeared_feature_mask are already defined

# Define the layers and colors
layers = [1, 4, 7, 10]
colors = ['darkred', 'orange', 'lightgreen', 'darkblue']

plt.figure(figsize=(12, 10))

for idx, layer in enumerate(layers):
    mask = disappeared_feature_mask[layer, :]
    x = avg_feat[layer, mask]
    y = avg_err_projection_onto_feat[layer, mask]
    
    # Remove NaNs
    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]
    
    plt.scatter(x, y, c=colors[idx], alpha=0.6, label=f'Layer {layer}')
    
    # Calculate R-squared for this layer
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2
    
    # Plot R-squared line for this layer
    line_x = np.array([-0.2, 25])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='black', linewidth=1, 
             label=f'R² Line Layer {layer} (R² = {r_squared:.3f})')
    
    # Add text label for the R² line
    mid_x = np.mean(line_x)
    mid_y = slope * mid_x + intercept
    plt.text(mid_x, mid_y, f'L{layer}', fontsize=10, ha='center', va='bottom', 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# Plot y=x line with increased visibility
plt.plot([-0.2, 25], [-0.2, 25], color='grey', linestyle='--', linewidth=2, label='y=x')

plt.xlim(-0.2, 25)
plt.ylim(-0.2, 16)
plt.xlabel('Feature Magnitude', fontsize=12)
plt.ylabel('Next Layer Error Projected Onto Prev Layer SAE W_dec', fontsize=12)
plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Scatter Plot of Feature Magnitude vs Projected Error', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming avg_err_projection_onto_feat, avg_feat, and disappeared_feature_mask are already defined

layers = [1, 4, 7, 10]
colors = ['darkred', 'orange', 'lightgreen', 'darkblue']

fig, ax = plt.subplots(figsize=(12, 8))

legend_elements = []
legend_labels = []

for idx, layer in enumerate(layers):
    mask = disappeared_feature_mask[layer, :]
    x = avg_feat[layer, mask]
    y = avg_err_projection_onto_feat[layer, mask]
    
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    
    scatter = ax.scatter(x, y, c=colors[idx], alpha=0.6)
    
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_squared = r_value**2
    
    line_x = np.array([-0.2, 25])
    line_y = slope * line_x + intercept
    line, = ax.plot(line_x, line_y, color='black', linewidth=1)
    
    legend_elements.append(scatter)
    legend_labels.append(f'Layer {layer} (R² = {r_squared:.3f})')

y_equals_x, = ax.plot([-0.2, 25], [-0.2, 25], color='grey', linestyle='--', linewidth=2)
legend_elements.append(y_equals_x)
legend_labels.append('y=x')

ax.set_xlim(-0.2, 25)
ax.set_ylim(-0.2, 16)
ax.set_xlabel('Feature Magnitude', fontsize=12)
ax.set_ylabel('Next Layer Error Projected Onto Prev Layer SAE W_dec', fontsize=12)
ax.grid(True, alpha=0.3)

# Create custom legend
legend = ax.legend(legend_elements, legend_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=5, fontsize=9, frameon=True, 
                   fancybox=True, borderpad=0.5, columnspacing=1)

# Adjust legend background
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('lightgray')
frame.set_linewidth(0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patheffects import withStroke

# Assuming avg_err_projection_onto_feat, avg_feat, and disappeared_feature_mask are already defined

layers = [1, 4, 7, 10]
colors = ['darkred', 'orange', 'lightgreen', 'darkblue']

fig, ax = plt.subplots(figsize=(12, 8))

legend_elements = []
legend_labels = []

for idx, layer in enumerate(layers):
    mask = disappeared_feature_mask[layer, :]
    x = avg_feat[layer, mask]
    y = avg_err_projection_onto_feat[layer, mask]
    
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    
    scatter = ax.scatter(x, y, c=colors[idx], alpha=0.6)
    
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_squared = r_value**2
    
    line_x = np.array([-0.2, 25])
    line_y = slope * line_x + intercept
    line, = ax.plot(line_x, line_y, color='black', linewidth=1)
    
    # Add label for R² line
    mid_x = np.mean(line_x)
    mid_y = slope * mid_x + intercept
    ax.text(mid_x, mid_y, f'L{layer}', color='black', fontsize=10, ha='center', va='bottom',
            path_effects=[withStroke(linewidth=3, foreground='lightgrey')])
    
    legend_elements.append(scatter)
    legend_labels.append(f'Layer {layer} (R² = {r_squared:.3f})')

# Plot y=x line with white border
y_equals_x, = ax.plot([-0.2, 25], [-0.2, 25], color='grey', linestyle='--', linewidth=2, 
                      path_effects=[withStroke(linewidth=4, foreground='white')])
legend_elements.append(y_equals_x)
legend_labels.append('y=x')

ax.set_xlim(-0.2, 25)
ax.set_ylim(-0.2, 16)
ax.set_xlabel('Feature Magnitude', fontsize=12)
ax.set_ylabel('Next Layer Error Projected Onto Prev Layer SAE W_dec', fontsize=12)
ax.grid(True, alpha=0.3)

# Create custom legend
legend = ax.legend(legend_elements, legend_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=5, fontsize=9, frameon=True, 
                   fancybox=True, borderpad=0.5, columnspacing=1)

# Adjust legend background
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('lightgray')
frame.set_linewidth(0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patheffects import withStroke

# Assuming avg_err_projection_onto_feat, avg_feat, and disappeared_feature_mask are already defined

layers = [1, 4, 7, 10]
colors = ['darkred', 'orange', 'lightgreen', 'darkblue']

fig, ax = plt.subplots(figsize=(12, 8))

legend_elements = []
legend_labels = []

for idx, layer in enumerate(layers):
    mask = disappeared_feature_mask[layer, :]
    x = avg_feat[layer, mask]
    y = avg_err_projection_onto_feat[layer, mask]
    
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]
    
    scatter = ax.scatter(x, y, c=colors[idx], alpha=0.6)
    
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_squared = r_value**2
    
    line_x = np.array([-0.2, 25])
    line_y = slope * line_x + intercept
    line, = ax.plot(line_x, line_y, color='black', linewidth=1)
    
    # Add label for R² line with grey box
    mid_x = np.mean(line_x)
    mid_y = slope * mid_x + intercept
    ax.text(mid_x, mid_y, f'L{layer}', color='black', fontsize=10, ha='center', va='center',
            bbox=dict(facecolor='lightgrey', edgecolor='none', alpha=0.7, pad=3))
    
    legend_elements.append(scatter)
    legend_labels.append(f'Layer {layer} (R² = {r_squared:.3f})')

# Plot y=x line with white border
y_equals_x, = ax.plot([-0.2, 25], [-0.2, 25], color='grey', linestyle='--', linewidth=2, 
                      path_effects=[withStroke(linewidth=4, foreground='white')])
legend_elements.append(y_equals_x)
legend_labels.append('y=x')

ax.set_xlim(-0.2, 25)
ax.set_ylim(-0.2, 16)
ax.set_xlabel('Feature Magnitude', fontsize=12)
ax.set_ylabel('Next Layer Error Projected Onto Prev Layer SAE W_dec', fontsize=12)
ax.grid(True, alpha=0.3)

# Create custom legend
legend = ax.legend(legend_elements, legend_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=5, fontsize=9, frameon=True, 
                   fancybox=True, borderpad=0.5, columnspacing=1)

# Adjust legend background
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('lightgray')
frame.set_linewidth(0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_scatter_with_r2(ax, x, y, idx):
    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    
    # Calculate 1st and 99th percentiles for x and y
    _, x_high = np.percentile(x, [1, 99])
    __, y_high = np.percentile(y, [1, 99])
    x_low = -0.2
    y_low = -0.2
    
    # Set limits
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.5, color='blue')
    
    # Calculate and plot regression line
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    x_reg = np.array([x_low, x_high])
    y_reg = slope * x_reg + intercept
    ax.plot(x_reg, y_reg, color='orange', label=f'R² = {r_value**2:.3f}',
            linewidth=3)
    
    # Plot y=x line
    ax.plot([max(x_low, y_low), min(x_high, y_high)], 
            [max(x_low, y_low), min(x_high, y_high)], 
            color='red', linestyle='--', label='y=x',linewidth=3)
    
    # Labels
    ax.set_xlabel(f'Layer {idx} Feature Magnitude')
    ax.set_ylabel(f'Layer {idx+1} Error Projected Onto Layer {idx} SAE W_dec')
    ax.legend()

# Assuming avg_err_projection_onto_feat, avg_feat, and disappeared_feature_mask are defined

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

layer_idxes = [1, 4, 7, 10]

for i, idx in enumerate(layer_idxes):
    x = avg_feat[idx, disappeared_feature_mask[idx, :]]
    y = avg_err_projection_onto_feat[idx, disappeared_feature_mask[idx, :]]
    plot_scatter_with_r2(axs[i], x, y, idx)

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_scatter_with_r2(ax, x, y, idx):
    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    
    # Calculate 1st and 99th percentiles for x and y
    _, x_high = np.percentile(x, [1, 99])
    __, y_high = np.percentile(y, [1, 99])
    x_low = -0.2
    y_low = -0.2
    
    # Set limits
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.5, color='blue')
    
    # Calculate and plot regression line
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    x_reg = np.array([x_low, x_high])
    y_reg = slope * x_reg + intercept
    ax.plot(x_reg, y_reg, color='black', label=f'R² = {r_value**2:.3f}')
    
    # Plot y=x line
    ax.plot([max(x_low, y_low), min(x_high, y_high)], 
            [max(x_low, y_low), min(x_high, y_high)], 
            color='red', linestyle='--', label='y=x')
    
    # Labels with larger font
    ax.set_xlabel(f'Layer {idx} Feature Magnitude', fontsize=16)  # Increased font size
    ax.set_ylabel(f'Layer {idx+1} Error Projected Onto Layer {idx} SAE W_dec', fontsize=16)  # Increased font size
    ax.legend(fontsize=14)  # Increased font size
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increased font size

# Assuming avg_err_projection_onto_feat, avg_feat, and disappeared_feature_mask are defined

plt.rcParams.update({'font.size': 20})  # Increase the default font size

fig, axs = plt.subplots(2, 2, figsize=(20, 20))  # Increased figure size for better readability
axs = axs.ravel()

layer_idxes = [1, 4, 7, 10]

for i, idx in enumerate(layer_idxes):
    x = avg_feat[idx, disappeared_feature_mask[idx, :]]
    y = avg_err_projection_onto_feat[idx, disappeared_feature_mask[idx, :]]
    plot_scatter_with_r2(axs[i], x, y, idx)

plt.tight_layout()
plt.show()
# %%
