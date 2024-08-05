# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt


# %%
def compare_measures(measure_1, measure_2, layer, number_of_samples, log=False, d_sae=24576):
    artefacts_folder = f'../../../artefacts/similarity_measures'
    layer = 0
    d_sae = 24576

    filename_1 = f'{artefacts_folder}/{measure_1}/.unclamped/res_jb_sae_feature_similarity_{measure_1}_1M_0.0_{layer}.npz'
    filename_2 = f'{artefacts_folder}/{measure_2}/.unclamped/res_jb_sae_feature_similarity_{measure_2}_1M_0.0_{layer}.npz'

    similarities_1 = np.load(filename_1)['arr_0']
    similarities_2 = np.load(filename_2)['arr_0']

    samples = np.random.randint(0, d_sae, size=(2, number_of_samples))

    score_1 = similarities_1[samples[0], samples[1]]
    score_2 = similarities_2[samples[0], samples[1]]

    # Filter out indices where at least one of the scores is invalid
    if log:
        valid_indices = np.logical_and(score_1 > 0, score_2 > 0)
    else: 
        valid_indices = np.logical_and(~np.isnan(score_1), ~np.isnan(score_2))
        
    print(f'{valid_indices.sum()}/{number_of_samples} samples are valid.')

    score_1 = score_1[valid_indices]
    score_2 = score_2[valid_indices]

    plt.scatter(score_1, score_2)
    plt.xlabel(measure_1)
    plt.ylabel(measure_2)
    
    if log:
        plt.xscale('log')
        plt.yscale('log')

        slope, intercept = np.polyfit(np.log(score_1), np.log(score_2), 1)

        # Calculate the regression line
        regression_x = np.logspace(-7, 0, 8)
        regression_y = np.exp(intercept) * regression_x ** slope
        regression_label = f'Regression line: y = {np.exp(intercept):.2f} * x^{slope:.2f}'
    else:
        slope, intercept = np.polyfit(score_1, score_2, 1)
        
        regression_x = np.linspace(0, 1, 10)
        regression_y = intercept + regression_x * slope
        regression_label = f'Regression line: y = {intercept:.2f} + x*{slope:.2f}'

    # Plot the regression line
    plt.plot(regression_x, regression_y, color='red', label=regression_label)

    plt.title(f'Comparison of similarity measures (layer {layer})')
    plt.legend()
    plt.show()


# %%
compare_measures('pearson_correlation', 'jaccard_similarity', layer=1, number_of_samples=100000, log=False)
compare_measures('pearson_correlation', 'mutual_information', layer=1, number_of_samples=100000, log=False)
compare_measures('pearson_correlation', 'sufficiency', layer=1, number_of_samples=100000, log=False)
compare_measures('pearson_correlation', 'necessity', layer=1, number_of_samples=100000, log=False)