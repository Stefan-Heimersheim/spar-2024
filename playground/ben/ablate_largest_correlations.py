# %%
from experiments.ablate_single_feature import AblationAggregator
import numpy as np
import torch as t
import argparse

# %%
d_sae = 24576
num_layers = 2

num_top_feats_to_ablate_per_layer = 1
num_batches = 2
batch_size = 2

# %%
top_corr_flat_idx = np.load("artefacts/ablations/top_1000_pearson_per_layer_flattened_feature_idx.npy")
# for all high-value pearson correlations, loading the first feature from each pair in a (num_layers, num_top_features) matrix
prev_layer_feat_idxes, _ = (
    np.array([
        np.unravel_index(top_corr_flat_idx[layer_idx], shape=(d_sae, d_sae))[ordering_idx]
        for layer_idx in range(num_layers)
    ])
    for ordering_idx in range(2)
)

# %%
diff_agg = AblationAggregator(num_batches=num_batches, batch_size=batch_size)
# %%
for layer_idx in range(num_layers):
    for feat_idx_idx in range(num_top_feats_to_ablate_per_layer):
        feat_idx = prev_layer_feat_idxes[layer_idx][feat_idx_idx]
        diff_agg.aggregate(first_layer_idx=layer_idx, feature_idx=feat_idx)
        diff_agg.save()
# %%
