"""
given a similarity measure, will read the recorded values from artefacts/
and then create a new file which samples from those values and then creates a new, smaller
dataframe in ablations/

see here https://docs.google.com/document/d/1nTVtRB9qgXsNb0NtERamyvi8a9J8MH1Q2RH17PR3iDo/edit#bookmark=id.nd6kw9yrxctg

saves data in the form (num_layers, num_flattened_feature_idxes), where each value in dim=1
represents a flattened idx (see interaction_scores.reshape() below to understand the flattening)
"""
# %% tb
import argparse
import numpy as np
from math import prod
from src.enums import Measure
from joblib import Parallel, delayed


from dataclasses import dataclass

@dataclass
class Args:
    measure: Measure = Measure.pearson
    truncate: bool = True
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
flattened_per_layer_scores = interaction_scores.reshape(num_layers, d_sae**2)
if args.truncate:
    flattened_per_layer_scores = flattened_per_layer_scores[:,:10000000]

# key line, very slow
print("Sorting each layer")
# the indexes of the layer pairs, per layer, sorted by the score of each pair, descending
sorted_desc_by_score_per_layer_pair_idxes = np.argsort(flattened_per_layer_scores, axis=1)[::-1]

# %%
layer_indices = np.arange(num_layers)[:, np.newaxis]

num_pairs_to_sample = 1000
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