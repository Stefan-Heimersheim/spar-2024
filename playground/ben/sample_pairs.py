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
from src.enums import Measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--measure', type=Measure, help='interaction measure whose values will indicate the', default=Measure.pearson)
    parser.add_argument('--truncate', type=bool, help='truncates the number of feature pairs that are sampled, used only for testing bc its fast', default=False)
    parser.add_argument('--save', type=bool, help='whether to save results or not', default=False)
    args = parser.parse_args()
    
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

    flattened_per_layer_scores = interaction_scores.reshape(num_layers, d_sae**2)
    if args.truncate:
        flattened_per_layer_scores = flattened_per_layer_scores[:,:10000000]
    
    # key line, very slow
    print("Sorting each layer")
    sorted_by_score_per_layer_pair_idxes = np.argsort(flattened_per_layer_scores, axis=1)

    layer_indices = np.arange(num_layers)[:, np.newaxis]

    num_pairs_to_sample = 1000
    _, curr_num_pairs_per_layer =  sorted_by_score_per_layer_pair_idxes.shape
    lower_bound_idx = int(curr_num_pairs_per_layer * 0.2)
    step_size = (curr_num_pairs_per_layer - lower_bound_idx) // num_pairs_to_sample

    # getting evenly spaced samples, as per https://docs.google.com/document/d/1nTVtRB9qgXsNb0NtERamyvi8a9J8MH1Q2RH17PR3iDo/edit#bookmark=id.nd6kw9yrxctg
    
    output_filename = f"artefacts/sampled_interaction_measures/{args.measure.value}/count_{num_pairs_to_sample}" 

    if args.save:
        print(f"Saving sampled pairs per layer")
        np.save(
            output_filename,
            sorted_by_score_per_layer_pair_idxes[:, lower_bound_idx::step_size]
        )
# %%
