# %%
from experiments.ablations.ablate_single_feature import AblationAggregator
import numpy as np
import torch as t
import argparse
from src.enums import Measure

def get_correlations_idxes(measure: Measure):
    return f"artefacts/ablations/top_1000_{measure.value}_per_layer_flattened_feature_idx.npy"
# %%
d_sae = 24576

# %%
def get_prev_layer_feat_idxes(measure: Measure):
    top_corr_flat_idx = np.load(measure.value)
    # for all high-value pearson correlations, loading the first feature from each pair in a (num_layers, num_top_features) matrix
    prev_layer_feat_idxes, _ = (
        np.array([
            np.unravel_index(top_corr_flat_idx[layer_idx], shape=(d_sae, d_sae))[ordering_idx]
            for layer_idx in range(top_corr_flat_idx.shape[0])
        ])
        for ordering_idx in range(2)
    )
    return prev_layer_feat_idxes

# %%
def main(args: argparse.Namespace):
    prev_layer_feat_idxes = get_prev_layer_feat_idxes(args.measure)
    # ablate
    diff_agg = AblationAggregator(num_batches=args.nb, batch_size=args.bs)
    # collect for top 100 and random 100 (which we assume will be low, make sure to dedupe)
    for layer_idx in range(args.first_layer, args.last_layer):
        for feat_idx_idx in range(args.num_feats):
            feat_idx = prev_layer_feat_idxes[layer_idx][feat_idx_idx]
            diff_agg.aggregate(first_layer_idx=layer_idx, first_feature_idx=feat_idx)
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