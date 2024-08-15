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
import numpy as np
from src.enums import Measure
from dataclasses import dataclass

@dataclass
class Args:
    measure: Measure = Measure.jaccard
    samples_per_bin: int = 10
    save: bool = True
rng = np.random.default_rng()
args = Args()
# %%
print("Reading full interaction scores")
if args.measure == Measure.pearson:
    filename = "artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz"
    with open(filename, 'rb') as data:
        interaction_scores = np.load(data)['arr_0']
    bins = np.linspace(0.01, 1.0, 10)
elif args.measure == Measure.jaccard:
    filename = "artefacts/jaccard/jaccard_2024-07-02-a.npz"
    with open(filename, 'rb') as data:
        interaction_scores = np.load(data)['arr_0']
    bins = np.linspace(0.01, 1.0, 10)
else:
    raise NotImplementedError(f"Haven't gotten around to {args.measure} yet")
num_layers, d_sae, _ = interaction_scores.shape

# %% sample across all layers
all_layer_sampled_pairs_arr = []
for layer_idx in range(num_layers):
    one_layer_tensor= interaction_scores[layer_idx]

    xs = []
    ys = []

    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i+1]
        if lower <= 0 <= upper:
            continue
        
        # Find all indices where the value is in the current bin
        indices = np.where((one_layer_tensor >= lower) & (one_layer_tensor < upper))
        num_pairs = len(indices[0])
        
        # If we have more valid pairs than we need, randomly sample
        if num_pairs > args.samples_per_bin:
            random_idx_idxes = rng.choice(a=num_pairs, size=args.samples_per_bin, replace=False)
            sampled_x = indices[0][random_idx_idxes]
            sampled_y = indices[1][random_idx_idxes]
        else:
            # If we don't have enough, take all of them
            sampled_x, sampled_y = indices
        xs.extend(sampled_x)
        ys.extend(sampled_y)
    one_layer_sampled_pairs = np.array([xs, ys]).T
    all_layer_sampled_pairs_arr.append(one_layer_sampled_pairs)
all_layer_sampled_pairs = np.array(all_layer_sampled_pairs_arr)
# %%
_, num_pairs_per_layer, __ = all_layer_sampled_pairs.shape 
output_filename = f"artefacts/sampled_interaction_measures/{args.measure.value}/count_{num_pairs_per_layer}" 
if args.save:
    print(
        """
        Saving sampled pairs per layer: (num_layers, num_feature_pairs, layer_idx_of_feat).
        data[1][32][0] = 42, data[1][32][1] = 8323
        across all the pairs of features where the first feature comes from layer 1
        the ~32 percentile feature pair
        is Layer 1 SAE feat 42, Layer 2 SAE feature 8323
        """
    )
    np.savez(output_filename, all_layer_sampled_pairs)
# %%
