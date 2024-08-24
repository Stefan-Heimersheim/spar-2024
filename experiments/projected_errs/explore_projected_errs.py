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
data = np.load("artefacts/projected_errs/total_num_toks__2097152.npz")
projection_sums = data["projection_sums"]
counts = data['counts'] 
ratio = projection_sums / counts

# %%
# as per https://spar2024.slack.com/archives/C0794GNT8KS/p1724432685121449?thread_ts=1724427023.459439&cid=C0794GNT8KS
# need to find the length of the feature directions to compare
def get_decoder_tensor() -> t.Tensor:
    print("Loading SAE decoder tensor")
    decoder_arr = []
    for layer in tqdm(list(range(num_layers))):
        sae_id = f"blocks.{layer}.hook_resid_pre"
        sae, _, _ = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=sae_id,
            device=device
        )
        sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
        decoder_arr.append(sae.W_dec)
    decoders = t.stack(decoder_arr, dim=0)
    return decoders
decoder_weights = get_decoder_tensor()
# %%
layer_data = ratio[0]

plt.figure(figsize=(12, 6))

# Create the histogram and KDE plot
sns.histplot(layer_data, kde=True, stat="density", kde_kws={"bw_adjust": 0.5})

plt.title("Distribution of SAE Feature Projections (Layer 0)")
plt.xlabel("Projection Magnitude")
plt.ylabel("Density")

# Add vertical lines for some summary statistics
plt.axvline(layer_data.mean(), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {layer_data.mean():.2f}')
plt.axvline(np.median(layer_data), color='g', linestyle='dashed', linewidth=1, label=f'Median: {np.median(layer_data):.2f}')

plt.legend()
plt.tight_layout()
plt.show()
# %%
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Distribution of SAE Feature Projections by Layer", fontsize=16)

def kde_area_proportion(x, y, lower, upper):
    mask = (x >= lower) & (x <= upper)
    total_area = integrate.trapz(y, x)
    section_area = integrate.trapz(y[mask], x[mask])
    return section_area / total_area

for i, ax in enumerate(axs.flatten()):
    if i < ratio.shape[0]:
        layer_data = ratio[i]
        
        # Calculate KDE
        kde = sns.kdeplot(layer_data, ax=ax, color='blue')
        
        # Get the KDE curve data
        line = kde.lines[0]
        x, y = line.get_data()
        
        # Calculate the area proportion
        proportion = kde_area_proportion(x, y, 0.9, 1.1)
        
        # Color the section between 0.9 and 1.1
        ax.fill_between(x, y, where=(x >= 0.9) & (x <= 1.1), color='yellow', alpha=0.5)
        
        # Add text for proportion
        ax.text(1.0, ax.get_ylim()[1], f'{proportion:.3f}', 
                horizontalalignment='center', verticalalignment='bottom')
        
        ax.set_title(f"Layer {i}")
        ax.set_xlim(0, 2)  # Adjust this range if needed
    else:
        ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
# %%
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Distribution of SAE Feature Projections by Layer", fontsize=16)

def kde_area_proportion(x, y, lower, upper):
    mask = (x >= lower) & (x <= upper)
    total_area = integrate.trapz(y, x)
    section_area = integrate.trapz(y[mask], x[mask])
    return section_area / total_area

for i, ax in enumerate(axs.flatten()):
    if i < ratio.shape[0]:
        layer_data = ratio[i]
        
        # Remove NaN values
        layer_data = layer_data[~np.isnan(layer_data)]
        
        if len(layer_data) > 0:  # Check if there's any data left after removing NaNs
            # Calculate KDE
            kde = sns.kdeplot(layer_data, ax=ax, color='blue')
            
            # Get the KDE curve data
            line = kde.lines[0]
            x, y = line.get_data()
            
            # Calculate the area proportion
            proportion = kde_area_proportion(x, y, 0.9, 1.1)
            
            # Color the section between 0.9 and 1.1
            ax.fill_between(x, y, where=(x >= 0.9) & (x <= 1.1), color='yellow', alpha=0.5)
            
            # Find the y-value at x=1.0 for text placement
            idx = np.argmin(np.abs(x - 1.0))
            y_text = y[idx]
            
            # Add text for proportion on top of the curve
            ax.text(1.0, y_text, f'{proportion:.3f}', 
                    horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
            ax.set_title(f"Layer {i}")
            ax.set_xlim(0, 2)  # Adjust this range if needed
        else:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
            ax.set_title(f"Layer {i}")
    else:
        ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
# %%
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import stats

# Assuming your matrix is named 'data' and has shape (11, 24560)
# If not, replace 'data' with your actual variable name
data = ratio

def calculate_area_percentage(kde, lower, upper):
    # Calculate total area
    total_area, _ = integrate.quad(kde.pdf, kde.dataset.min(), kde.dataset.max())
    
    # Calculate area between lower and upper bounds
    area_between, _ = integrate.quad(kde.pdf, lower, upper)
    
    # Calculate percentage
    percentage = (area_between / total_area) * 100
    return percentage

# Create a 4x3 grid of subplots (11 in total, last one will be empty)
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
axs = axs.flatten()

for i in range(11):
    # Get data for current layer and remove NaN values
    layer_data = data[i]
    layer_data = layer_data[~np.isnan(layer_data)]
    
    if len(layer_data) == 0:
        axs[i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
        axs[i].set_title(f'Layer {i+1}')
        continue
    
    # Create KDE plot
    sns.kdeplot(layer_data, ax=axs[i])
    
    # Calculate percentage of area between 0.9 and 1.1
    kde = stats.gaussian_kde(layer_data)
    percentage = calculate_area_percentage(kde, 0.9, 1.1)
    
    # Add annotation
    axs[i].annotate(f'{percentage:.2f}% area\nbetween 0.9-1.1', 
                    xy=(0.7, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    axs[i].set_title(f'Layer {i+1}')
    axs[i].set_xlabel('Average Magnitude')
    axs[i].set_ylabel('Density')

# Remove the last (empty) subplot
fig.delaxes(axs[11])

plt.tight_layout()
plt.show()
# %%
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate

# Assuming your matrix is named 'data' and has shape (11, 24560)
# If not, replace 'data' with your actual variable name
data = ratio

def calculate_area_percentage(kde, lower, upper):
    # Calculate total area
    total_area, _ = integrate.quad(kde.pdf, kde.dataset.min(), kde.dataset.max())
    
    # Calculate area between lower and upper bounds
    area_between, _ = integrate.quad(kde.pdf, lower, upper)
    
    # Calculate percentage
    percentage = (area_between / total_area) * 100
    return percentage

def find_99_percent_range(kde):
    def area_to_99(x):
        return calculate_area_percentage(kde, -x, x) - 99

    from scipy.optimize import brentq
    x_99 = brentq(area_to_99, 0, 1000)  # Adjust upper bound if necessary
    return -x_99, x_99

# Create a 4x3 grid of subplots (11 in total, last one will be empty)
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
axs = axs.flatten()

for i in range(11):
    # Get data for current layer and remove NaN values
    layer_data = data[i]
    layer_data = layer_data[~np.isnan(layer_data)]
    
    if len(layer_data) == 0:
        axs[i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
        axs[i].set_title(f'Layer {i+1}')
        continue
    
    # Create KDE
    kde = stats.gaussian_kde(layer_data)
    
    # Find range that covers 99% of the area
    x_min, x_max = find_99_percent_range(kde)
    
    # Create KDE plot with adjusted x-range
    x = np.linspace(x_min, x_max, 1000)
    axs[i].plot(x, kde(x))
    axs[i].set_xlim(x_min, x_max)
    
    # Calculate percentage of area between 0.9 and 1.1
    percentage = calculate_area_percentage(kde, 0.9, 1.1)
    
    # Add annotation
    axs[i].annotate(f'{percentage:.2f}% area\nbetween 0.9-1.1', 
                    xy=(0.7, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    axs[i].set_title(f'Layer {i+1}')
    axs[i].set_xlabel('Average Magnitude')
    axs[i].set_ylabel('Density')

# Remove the last (empty) subplot
fig.delaxes(axs[11])

plt.tight_layout()
plt.show()

# %%
