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
sae_name = 'res_jb_sae'
n_layers = 12

measure_name = "pearson_correlation"
activation_threshold = None

folder = f'../../artefacts/similarity_measures/{measure_name}/.unclamped'
similarities_centered = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens="10M", first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])

measure_name = "activation_cosine_similarity"
activation_threshold = 0.2

folder = f'../../artefacts/similarity_measures/{measure_name}/.unclamped'
similarities_uncentered = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens="10M", first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])


# %%
# Analyze difference of these two matrices with respect to NaN entries
nan_centered = np.isnan(similarities_centered)
nan_uncentered = np.isnan(similarities_uncentered)

both_nan = np.sum(nan_centered & nan_uncentered)
only_arr1_nan = np.sum(nan_centered & ~nan_uncentered)
only_arr2_nan = np.sum(~nan_centered & nan_uncentered)
neither_nan = np.sum(~nan_centered & ~nan_uncentered)

confusion_matrix = np.array([[neither_nan, only_arr2_nan], [only_arr1_nan, both_nan]])


def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues):
    percentages = cm.astype('float') / cm.sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix of NaN values for centered and uncentered cosine similarity',
           ylabel='Centered cosine similarity (Pearson correlation)',
           xlabel='Uncentered cosine similarity')

    plt.setp(ax.get_xticklabels(), ha="right")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,} ({percentages[i, j]:.1%})',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax


plot_confusion_matrix(confusion_matrix, classes=['Not NaN', 'NaN'])
plt.show()


# %%
# Plot histogram of absolute difference of entries
diff_flat = np.abs(similarities_centered - similarities_uncentered)[~nan_centered & ~nan_uncentered]

num_bins = 200

fig, ax = plt.subplots(figsize=(8, 8))
ax.hist(diff_flat, bins=num_bins, range=(0, 2))

ax.set_title(f"Histogram of absolute differences between similarity matrices")
ax.set_xlabel("Absolute difference")
ax.set_ylabel("Number of feature pairs")

ax.set_xlim(0, 0.5)
ax.set_yscale('log')

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

line_x = diff_flat.mean()
plt.axvline(x=line_x, color='red', linestyle='--', linewidth=3)

# Add the label
plt.text(line_x + 0.02, plt.ylim()[1] / 2, f'Mean: {line_x:.5f}', 
         horizontalalignment='left',
         verticalalignment='top',
         rotation=90,
         color='red')

plt.show()
