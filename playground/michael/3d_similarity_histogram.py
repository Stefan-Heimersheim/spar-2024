# %%
import os
import numpy as np
import sys
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.ticker import LinearLocator

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import get_filename, load_similarity_data


# %%
# Load similarity matrices
measure_name = "pearson_correlation"
sae_name = 'res_jb_sae'
n_layers = 12
activation_threshold = None
n_tokens = '10M'

folder = f'../../artefacts/similarity_measures/{measure_name}/.unclamped'

similarities = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens=n_tokens, first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])


# %%
thresholds = np.linspace(0, 1, 21)
high_sim_features = np.arange(15)


# %%
data = []
for threshold in tqdm(thresholds):
    pass_through_connections = (similarities >= threshold)
    n_forward_pass_through = pass_through_connections.sum(axis=-1).flatten()

    data.append([(n_forward_pass_through >= n).sum() for n in high_sim_features] + [(n_forward_pass_through > max(high_sim_features)).sum()])

data = np.array(data)

n_thresholds, n_high_sim_features = data.shape


# %%
from matplotlib.colors import LogNorm
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='viridis', norm=LogNorm(vmin=1, vmax=data.max()))

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Number of features (log scale)', rotation=-90, va="bottom")

# Set title and labels
plt.title('Pearson correlation: High-similarity downstream neighbors')
plt.xlabel('Number of high-similarity downstream neighbors')
ax.set_xticks(range(n_high_sim_features))
ax.set_xticklabels([str(n) if n % 3 == 0 else '' for n in high_sim_features] + [f'> {max(high_sim_features)}'])

plt.ylabel('High-similarity threshold')
y_ticks = range(len(data))
y_labels = [f'{(y / len(data-1)):.1f}' if y % 4 == 0 else '' for y in y_ticks]  # Labels every 0.2

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Show the plot
plt.show()
