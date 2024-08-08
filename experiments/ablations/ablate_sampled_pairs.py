# %%
from experiments.ablations.ablate_single_feature import AblationAggregator
import numpy as np
import torch as t
import argparse
from src.enums import Measure
from collections import defaultdict
from dataclasses import dataclass

def get_correlations_idxes(measure: Measure):
    return np.load(f"artefacts/sampled_interaction_measures/{measure.value}/count_75.npz")['arr_0']
# %%
@dataclass
class Args:
    measure = Measure.pearson
    nb = 4
    bs = 128
    first_layer = 0
    last_layer = 2
# TODO: rename
args = Args()
diff_agg = AblationAggregator(num_batches=2, batch_size=args.bs)
corr_idxes = get_correlations_idxes(args.measure)
for layer_idx in range(args.first_layer, args.last_layer):
    prev_feat_to_next_feats = defaultdict(list) 
    for prev_feat, next_feat in corr_idxes[layer_idx, :, :]:
        prev_feat_to_next_feats[prev_feat].append(next_feat)
    for prev_feat, next_feats in prev_feat_to_next_feats.items():
        diff_agg.aggregate(
            prev_layer_idx=layer_idx,
            prev_feat_idx=prev_feat,
        )
        masked_means = diff_agg.masked_means
        diff_agg.save(next_feature_idxes=next_feats)
        break
# %%
def main(args: argparse.Namespace):
    corr_idxes = get_correlations_idxes(args.measure)

    # ablate
    diff_agg = AblationAggregator(num_batches=args.nb, batch_size=args.bs)
    # collect for top 100 and random 100 (which we assume will be low, make sure to dedupe)
    for layer_idx in range(args.first_layer, args.last_layer):
        prev_feat_to_next_feats = defaultdict(list)
        for feat_idx_idx in range(args.num_feats):
            feat_idx = prev_layer_feat_idxes[layer_idx][feat_idx_idx]
            diff_agg.aggregate(prev_layer_idx=layer_idx, prev_feat_idx=feat_idx)
            if not args.dry_run:
                diff_agg.save() 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-layer', type=int, help='first layer idx', default=0)
    parser.add_argument('--last-layer', type=int, help='last layer idx (exclusive)', default=1, )
    parser.add_argument('--measure', type=Measure, help='interaction measure whose values will indicate the', default=1, )
    parser.add_argument('--nb', type=int, help='num of batches', default=2)
    parser.add_argument('--bs', type=int, help='batch size', default=2)
    parser.add_argument('--num-feats', type=int, help='number of features per row, starting from the order in the data file, to ablate', default=1)
    parser.add_argument('--dry-run', type=bool, help='dry run (do not save)', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.first_layer >= args.last_layer:
        parser.error(f"Argument {args.first_layer=} must be less than {args.last_layer=}")
    main(args)