# Artefacts
This folder contains all (important) artefacts created during the experiments, and they generally correspond to the scripts in the respective `/experiments` sub-folders. Whenever possible, we use the compressed `numpy` data format (`.npz`) to save disk space.

## Maximum activations of SAE features (`max_sae_activations`)
SAE features have vastly different activation ranges. The file [res_jb_max_sae_activations_17.5M.npz](/artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M.npz) contains the maximum activation per SAE feature for all layers over 17.5M tokens, i.e., an array of shape `(n_layers, n_features) = (12, 24576)`.

Below, we show a histogram of maximum activations. The highest maxmimum activation of any feature is 639, but most features have maxmimum activations below 100.

![Histogram of maximum SAE feature activations](/artefacts/max_sae_activations/res_jb_max_sae_activations_17.5M_histogram.png)

## Dead features
If a feature is never (or very rarely) active in the dataset, any similarities with other features need to be ignored. The file [res_jb_sae_dead_features_17.5M_0.0.npz](/artefacts/dead_features/res_jb_sae_dead_features_17.5M_0.0.npz) contains the number of dead features per layer, evaluated in steps of 65k tokens for different "death thresholds". Given that the thresholds are defined as `[0, 5, 10, 50, 100, 500, 1000, 5000, 10000]`, we get an an array of shape `(n_steps, n_thresholds, n_layers) = (266, 9, 12)`.

Below, we show the number of dead features against the number of tokens. Most alive features activate after a few 100k tokens, but there is still some progress until the end.

![Number of dead SAE features](/artefacts/dead_features/res_jb_sae_dead_features_17.5M_0.0_0.png)

## Pair co-activation
Similar to counting non-zero feature activations to identify dead features, pair co-activation counts the number of tokens where both features of a pair are active. 

The file [res_jb_sae_pair_co_activation_17.5M_100_24576_0.0.npz](/artefacts/pair_co_activation/res_jb_sae_pair_co_activation_17.5M_100_24576_0.0.npz) contains the number of feature pairs which never (or rarely) co-activate, for all layer pairs and for different numbers of tokens. Again, there are multiple thresholds (`[0, 5, 10, 50, 100]`), resulting in an array of shape `(n_thresholds, n_layer_pairs, n_steps) = (5, 11, 266)`.

Below, we show the number of non-co-activating feature pairs against the number of tokens. 

![Number of non-co-activating pairs of SAE features](/artefacts/pair_co_activation/res_jb_sae_pair_co_activation_17.5M_100_24576_0.0.png)

## Similarity measures


## Upstream explanations


## Ablations