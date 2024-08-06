"""
given a similarity measure, will read the recorded values from artefacts/
and then create a new file which samples from those values and then creates a new, smaller
dataframe in ablations/

see here https://docs.google.com/document/d/1nTVtRB9qgXsNb0NtERamyvi8a9J8MH1Q2RH17PR3iDo/edit#bookmark=id.nd6kw9yrxctg

saves data in the form (num_layers, num_flattened_feature_idxes), where each value in dim=1
represents a flattened idx (see interaction_scores.reshape() below to understand the flattening)
# TODO: rewrite after it's amended
"""
# %% tb
import argparse
import numpy as np
from math import prod
from src.enums import Measure
import os
from dataclasses import dataclass
from typing import Dict
@dataclass
class Args:
    measure: Measure = Measure.pearson
    save: bool = False

# %%
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--measure', type=Measure, help='interaction measure whose values will indicate the', default=Measure.pearson)
#     parser.add_argument('--truncate', type=bool, help='truncates the number of feature pairs that are sampled, used only for testing bc its fast', default=False)
#     parser.add_argument('--save', type=bool, help='whether to save results or not', default=False)
#     args = parser.parse_args()


args = Args()
print("Reading full interaction scores")
if args.measure == Measure.pearson:
    filename = "artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz"
    with open(filename, 'rb') as data:
        interaction_scores = np.load(data)['arr_0']
elif args.measure == Measure.jaccard:
    filename = "artefacts/jaccard/jaccard_2024-07-02-a.npz"
    with open(filename, 'rb') as data:
        interaction_scores = np.load(data)['arr_0']
else:
    raise NotImplementedError(f"Haven't gotten around to {args.measure} yet")
num_layers, d_sae, _ = interaction_scores.shape

# %%
def convert_only_nonzero_pair_idxes_to_padded_arr(only_nonzero_pair_idxes: Dict):
    """turns a ragged dictionary of lists of indexes (all the nonzero pairs for that layer)
    into a nice rectangular matrix where shorter rows are padded to the length of the longest
    with nans

    Args:
        only_nonzero_pair_idxes (Dict): {"layer_idx": all_nonzero_flattened_pair_idxes_for_this_layer}
    """
    max_num_nonzeros = max(len(val) for val in only_nonzero_pair_idxes.values())
    return np.array(
        [
            np.pad(
                only_nonzero_pair_idxes[f'layer_{layer_idx}'].astype(float), 
                (0, max_num_nonzeros - len(only_nonzero_pair_idxes[f'layer_{layer_idx}'])), 
                mode='constant', 
                constant_values=np.nan
            ) 
            for layer_idx in range(num_layers)
        ]
    )

# %%
flattened_per_layer_scores = interaction_scores.reshape(num_layers, d_sae**2)
# the indexes of the layer pairs, per layer, sorted by the score of each pair, descending
sorted_flattened_pairs_filename = "artefacts/sampled_interaction_measures/pearson_correlation/sorted_nonzero_flattened_pair_idxes"
if os.path.exists(f"{sorted_flattened_pairs_filename}.npz"):
    only_nonzero_pair_idxes = np.load(f"{sorted_flattened_pairs_filename}.npz")
    padded_nonzero_sorted_desc_by_score_per_layer_pair_idxes = convert_only_nonzero_pair_idxes_to_padded_arr(only_nonzero_pair_idxes)
else:
    print("Sorting each layer")
    # key line, very slow
    full_sorted_desc_by_score_per_layer_pair_idxes = np.argsort(flattened_per_layer_scores, axis=1)[:, ::-1]
    per_layer_nonzero = np.count_nonzero(flattened_per_layer_scores, axis=1)
    only_nonzero_pair_idxes = {
        f"layer_{layer_idx}":
            full_sorted_desc_by_score_per_layer_pair_idxes[
                layer_idx,
                :per_layer_nonzero[layer_idx]
            ]
        for layer_idx in range(num_layers)
    }
    # saving so that I don't have to sort every time
    np.savez_compressed(sorted_flattened_pairs_filename, only_nonzero_pair_idxes)
    padded_nonzero_sorted_desc_by_score_per_layer_pair_idxes = convert_only_nonzero_pair_idxes_to_padded_arr(only_nonzero_pair_idxes)
# %%
num_pairs_to_sample_per_layer = 75
step_sizes = per_layer_nonzero // num_pairs_to_sample_per_layer
assert min(step_sizes) > 0, "count_nonzero < num_samples"
end_idxes = step_sizes * num_pairs_to_sample_per_layer
# getting evenly spaced samples, as per https://docs.google.com/document/d/1nTVtRB9qgXsNb0NtERamyvi8a9J8MH1Q2RH17PR3iDo/edit#bookmark=id.nd6kw9yrxctg
evenly_spaced_sampled_pair_idxes = np.array(
    [
        padded_nonzero_sorted_desc_by_score_per_layer_pair_idxes[layer, :end:step]
        for layer, (end, step) in enumerate(zip(end_idxes, step_sizes))
    ],
    dtype=int
)
# %%
prev_layer_feat_idx_arr = []
next_layer_feat_idx_arr = []
unflattened_pair_arr = []
for layer_idx in range(num_layers):
    unflattened_pair_arr.append(
        np.array(
            np.unravel_index(evenly_spaced_sampled_pair_idxes[layer_idx], (d_sae, d_sae))
        ).T
    )
unflattened_pairs = np.array(unflattened_pair_arr)

# %%
output_filename = f"artefacts/sampled_interaction_measures/{args.measure.value}/count_{num_pairs_to_sample_per_layer}" 
if args.save:
    print(
        """
        Saving sampled pairs per layer: (num_layers, num_feature_pairs, layer_idx_of_feat).
        data[1][32][0] = 42, data[1][32][1] = 8323
        across all the pairs of features where the first feature comes from layer 1
        the 32 percentile feature pair
        is Layer 1 SAE feat 42, Layer 2 SAE feature 8323
        """
    )
    np.savez(output_filename, unflattened_pairs)