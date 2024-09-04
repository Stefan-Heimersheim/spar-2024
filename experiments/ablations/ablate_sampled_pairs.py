# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'experiments'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from ablations.ablate_single_feature import AblationAggregator
import numpy as np
import torch as t
import argparse
from src.enums import Measure
from collections import defaultdict
from dataclasses import dataclass

while True:
    try:
        measure = getattr(Measure, input("Please enter measure value:"))
        break
    except AttributeError as e:
        print(e)

# %%
@dataclass
class Args:
    measure = measure
    nb = 128
    bs = 64
    count = 100
    first_layer = 0
    last_layer = 11
args = Args()
# %%
measure_to_filename = {
    Measure.pearson: f"artefacts/sampled_interaction_measures/{args.measure.value}/evenly_spaced_count_{args.count}.npz",
    Measure.jaccard: f"artefacts/sampled_interaction_measures/jaccard_similarity/count_100.npz",
    Measure.necessity: "artefacts/sampled_interaction_measures/necessity_relative_activation/count_100.npz",
    Measure.sufficiency: "artefacts/sampled_interaction_measures/sufficiency_relative_activation/count_100.npz",
}
corr_idxes = np.load(measure_to_filename[args.measure])['arr_0']
num_layers, num_feat_pairs, _ = corr_idxes.shape
mean_diff_results = np.zeros(shape=(num_layers, num_feat_pairs))

# %%
# TODO: rename
diff_agg = AblationAggregator(num_batches=args.nb, num_rows_per_batch=args.bs)
for layer_idx in range(args.first_layer, args.last_layer):
    prev_feat_to_next_feats = defaultdict(list) 
    prev_feat_to_idx_idx = {}
    # the idx within corr_idxes(dim=1)...not the VALUE, which is itself an idx...sorry this is confusing
    for prev_idx_idx, (prev_feat, next_feat) in enumerate(corr_idxes[layer_idx, :, :]):
        prev_feat_to_next_feats[prev_feat].append(next_feat)
        prev_feat_to_idx_idx[prev_feat] = prev_idx_idx
    for prev_feat, next_feats in prev_feat_to_next_feats.items():
        diff_agg.aggregate(
            prev_layer_idx=layer_idx,
            prev_feat_idx=prev_feat,
        )
        masked_mean_diffs = diff_agg.masked_mean_diffs
        prev_idx_idx = prev_feat_to_idx_idx[prev_feat]
        for next_feat in next_feats:
            mean_diff_results[layer_idx][prev_idx_idx] = (
                masked_mean_diffs[next_feat]  
            )
    filename = f"artefacts/ablations/{args.measure.value}/count_{args.count}__up_til_layer_{layer_idx}__toks_{args.nb * args.bs * 128}"
    print(f"Saving {filename}")
    np.savez(filename, mean_diff_results)
np.savez(f"artefacts/ablations/{args.measure.value}/count_{args.count}__final__toks_{args.nb * args.bs * 128}", mean_diff_results)
