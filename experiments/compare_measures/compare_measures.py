# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import itertools


# %%
def compare_similarity_measures(measure_1, bin_limits_1, measures_2, bin_limitss_2, layer, samples_per_bin):
    # Load similarity matrices
    artefacts_folder = f'../../artefacts/similarity_measures'
    sae_name = 'res_jb_sae'
    token_desc = '1M'
    activity_threshold = 0.0

    filename_1 = f'{artefacts_folder}/{measure_1}/.unclamped/{sae_name}_feature_similarity_{measure_1}_{token_desc}_{activity_threshold:.1f}_{layer}.npz'
    filenames_2 = [f'{artefacts_folder}/{measure_2}/.unclamped/{sae_name}_feature_similarity_{measure_2}_{token_desc}_{activity_threshold:.1f}_{layer}.npz' for measure_2 in measures_2]

    similarities_1 = np.load(filename_1)['arr_0'].flatten()
    similaritiess_2 = [np.load(filename_2)['arr_0'].flatten() for filename_2 in filenames_2]

    # Filter out indices where at least one measure is nan
    valid_indices = ~np.isnan(similarities_1)

    for sim_2 in similaritiess_2:
        valid_indices &= ~np.isnan(sim_2)

    # Create bins with indices by similarity values
    bins_1 = np.digitize(similarities_1, bin_limits_1)
    indices_1 = [np.where((bins_1 == i) & valid_indices)[0] for i in range(len(bin_limits_1 + 1))]

    binss_2 = [np.digitize(sim_2, bin_limits_2) for sim_2, bin_limits_2 in zip(similaritiess_2, bin_limitss_2)]
    indicess_2 = [[np.where((bins_2 == i) & valid_indices)[0] for i in range(len(bin_limits_2 + 1))] for bins_2, bin_limits_2 in zip(binss_2, bin_limitss_2)]

    # Sample indices from each bin for all measures
    indicess = itertools.chain(indices_1, *indicess_2)
    sample_indices = [i for indices in indicess for i in np.random.choice(indices, min(samples_per_bin, len(indices)))]

    selected_similarities_1 = similarities_1[sample_indices]
    selected_similaritiess_2 = [sim_2[sample_indices] for sim_2 in similaritiess_2]

    for measure_2, selected_sim_2 in zip(measures_2, selected_similaritiess_2):
        plt.scatter(selected_similarities_1, selected_sim_2, label=measure_2, s=0.1)
    
        slope, intercept = np.polyfit(selected_similarities_1, selected_sim_2, 1)

        regression_x = np.linspace(0, 1, 10)
        regression_y = intercept + regression_x * slope
        regression_label = f'Regression line: y = {intercept:.2f} + x*{slope:.2f}'

        # Plot the regression line
        plt.plot(regression_x, regression_y, label=regression_label, linewidth=2.0)

    plt.xlabel(measure_1)
    plt.ylabel(', '.join(measures_2))
    
    plt.title(f'Comparison of similarity measures (layer {layer}, {samples_per_bin} samples per bin)')
    plt.legend()
    plt.show()


# %%
compare_similarity_measures('pearson_correlation', np.arange(-0.9, 1.0, 0.1), ['jaccard_similarity', 'forward_implication'], [np.arange(0.1, 1.0, 0.1)] * 2, 0, 10000)


# %%
compare_similarity_measures('jaccard_similarity', np.arange(0.1, 1.0, 0.1), ['forward_implication', 'backward_implication'], [np.arange(0.1, 1.0, 0.1)] * 2, 0, 10000)
