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
import matplotlib.pyplot as plt
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
    padded_nonzero_sorted_desc_by_score_per_layer_pair_idxes =convert_only_nonzero_pair_idxes_to_padded_arr(only_nonzero_pair_idxes)
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
    np.savez_compressed(sorted_flattened_pairs_filename, only_nonzero_pair_idxes)
# %%
"""
1. sort all pairs in all layers
2. pick the top 50 pairs in all layers
3. in each layer, get the distinct first-layer features
4. for each of those first layer features, find the top second layer feature pairs
5. look at the histograms and collect the top pairs across each of the first layer pairs...so that I can run the ablations only on those

hmmm...is this too complicated?
how am I supposed to sample properly...
alright I guess I'll just try and go for it and see how far I get...

l0f14525 has only 5 nonzero values
array([11914, 15247, 18290, 22140, 22737])
[1.0000046, 1.0000031, 1.0000012, 1.0000013, 1.0000039]

okay I see 14525 is showing up 3 times at the top of first_layer_top_feat_idxes
14525, 14525, 19038,   663, 14525
so probably no good eh..?

TODO: should I keep these layer 0 features in? or not? I feel like I should skip the first 10? just to be safe??

"""

"""
Okay I think I want to count up the number of nonzeros for each feature idx...

okay what do I want?
I want evenly-spaced ranges
but what do I also want?
to minimize the amount of ablations that I need to do
I'm going to save like...freakin...NOT that much data...hmm...

okay what I really need is...what I REALLY need is...
the distribution of the actual values of the top 100 pairs
not just...ugh...

what I really care about is the even distribution of the weights
okay so I need to just save 100 evenly distributed pairs per layer
and just like...HOPE that they're all from some of the same upstream feature pairs
and just like...reduce the total number of pairs...and that'll make it fast
okay I'll make it like..75
"""

"""
Maybe I need to save the sorted idxes...let me see how big this would be

"""
# %%
per_layer_nonzero = np.count_nonzero(flattened_per_layer_scores, axis=1)
# %%
num_pairs_to_sample_per_layer = 75
_, curr_num_pairs_per_layer =  sorted_desc_by_score_per_layer_pair_idxes.shape
# get the nonzero percent on avg across all layers

nonzero_percent = np.count_nonzero(flattened_per_layer_scores) / prod(flattened_per_layer_scores.shape)

# try to only sample from the nonzero entries by having the same lower bound idx across all layers
# since they're sorted in descending order by score, this'll be the right bound
last_non_zero_idx = int(nonzero_percent * curr_num_pairs_per_layer)
assert last_non_zero_idx >= 1, "Not enough nonzero scores. Loosen the truncation"
step_size = last_non_zero_idx // num_pairs_to_sample
evenly_spaced_by_score_idxes = sorted_desc_by_score_per_layer_pair_idxes[:, :last_non_zero_idx:step_size]
# %%
# getting evenly spaced samples, as per https://docs.google.com/document/d/1nTVtRB9qgXsNb0NtERamyvi8a9J8MH1Q2RH17PR3iDo/edit#bookmark=id.nd6kw9yrxctg

output_filename = f"artefacts/sampled_interaction_measures/{args.measure.value}/count_{num_pairs_to_sample}" 

if args.save:
    print(f"Saving sampled pairs per layer")
    np.save(
        output_filename,
        sorted_desc_by_score_per_layer_pair_idxes[:, lower_bound_idx::step_size]
    )
# %%
"""okay so..
I need to figure out...how to evenly sample...
the issue is that they're going to be heavy tailed.
okay let me think about how to do this.
I think...it's just a matter of
we have n values
and then... k of them are 0
so we want to only look at (n-k) and sample across those
so i want to do nonzero counts of each layer
and then sample based on that
"""