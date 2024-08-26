# %%
import os
import numpy as np
import sys
import plotly.graph_objects as go
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import get_filename, load_similarity_data


# %%
# Load similarity matrices
measure_name = "pearson_correlation"
sae_name = 'res_jb_sae'
n_layers = 12
activation_threshold = None

folder = f'../../artefacts/similarity_measures/{measure_name}/.unclamped'

similarities_10 = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens="10M", first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])
similarities_100 = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens="100M", first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])


# %%
# Analyze difference of these two matrices
nan10 = np.isnan(similarities_10)
nan100 = np.isnan(similarities_100)

both_nan = np.sum(nan10 & nan100)
only_arr1_nan = np.sum(nan10 & ~nan100)
only_arr2_nan = np.sum(~nan10 & nan100)
neither_nan = np.sum(~nan10 & ~nan100)

confusion_matrix = np.array([[neither_nan, only_arr2_nan], [only_arr1_nan, both_nan]])


# %%
labels = ['Not NaN', 'NaN']
    
fig = go.Figure(data=go.Heatmap(
                z=confusion_matrix,
                x=labels,
                y=labels,
                hoverongaps=False,
                text=confusion_matrix,
                texttemplate="%{text}",
                colorscale='Blues'))

total = np.sum(confusion_matrix)
accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / total if total > 0 else 0

fig.update_layout(
    title=f'Confusion Matrix of Pearson Correlation NaN Entries (Accuracy: {accuracy:.1%})',
    font=dict(size=10),
    xaxis_title='100M tokens',
    yaxis_title='10M tokens')

fig.show()


# %%
diff_flat = np.abs(similarities_10 - similarities_100)[~nan10 & ~nan100]


 # %%
num_bins = 200

fig, ax = plt.subplots(figsize=(10, 6))
    
# Plot the histogram
ax.hist(diff_flat, bins=num_bins, range=(0, 2))

# Set titles and labels
ax.set_title(f"Histogram of Absolute Differences Between Similarity Matrices (Mean: {diff_flat.mean():.5f})")
ax.set_xlabel("Absolute Difference")
ax.set_ylabel("Frequency")

# Set x-axis range 
ax.set_xlim(0, 2)
ax.set_yscale('log')

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

plt.show()


# %%
n_zeros = (diff_flat <= 0.001).sum()
n_zeros / len(diff_flat)