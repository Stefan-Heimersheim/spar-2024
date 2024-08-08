# %%
import numpy as np


# %%
co_activation_thresholds = [0, 5, 10, 50, 100]
evaluation_frequency = 16 * 32 * 128
n_layers = 12
n_features_1 = 100
n_features_2 = 24576

data = np.load('../../artefacts/dead_feature_pairs/res_jb_sae_dead_feature_pairs_17.5M_100_24576_0.0.npz')['arr_0']
data = np.concatenate([np.ones((len(co_activation_thresholds), n_layers - 1, 1)) * n_features_1 * n_features_2, data], axis=-1)

# %%
data.shape

# %%
threshold_index = 2
layer = 4
# Layer 4/5 after 2M tokens
step = int(2 ** 21 / evaluation_frequency)

print(f'After 2M tokens (step {step}): {data[threshold_index, layer, step] / (n_features_1 * n_features_2) * 100:.2f}%')

# Layer 4/5 after 2M tokens
step = int(1e7 / evaluation_frequency)

print(f'After 10M tokens (step {step}): {data[threshold_index, layer, step] / (n_features_1 * n_features_2) * 100:.2f}%')

# Layer 4/5 after 17.5M tokens
step = -1

print(f'After 17.5M tokens (step {step}): {data[threshold_index, layer, step] / (n_features_1 * n_features_2) * 100:.2f}%')
# %%
