# spar-2024

## Folder structure
### Playground
This folder is for quick-and-dirty experimenting without direct applicability to the project results.

### Experiments


### Artefacts
#### Dead features ([Data](/artefacts/dead_features/res_jb_sae_dead_features_17.5M_0.0.npz), [Plot](/artefacts/dead_features/res_jb_sae_dead_features_17.5M_0.0_0.png))
##### Description
This file contains the number of dead SAE features across all layers for different amounts of data. Its shape is `(n_steps, n_thresholds, n_layers)` (= `(XX, 9, 12)`). 
- `n_steps` is the number of evaluations along the x-axis of the plot below. We're evaluating every 16 batches, i.e., every 16 * 4096 = 65536 tokens. Given that we have 17.5M tokens in total, this means that we have XX evaluations.
- `n_thresholds` is the number of thresholds for defining "dead" features. We're not only testing for 0 activations, but use the 9 thresholds 0, 5, 10, 50, 100, 500, 1,000, 5,000, and 10,000.
- `n_layers` is the number of layers in the model, i.e., the number of SAEs. We track the number of dead features for each layer individually.
##### Plot
As expected, the number of dead SAE features (defined as `threshold=0`) declines (slightly) when we use more tokens. The connection to the number of [dead feature pairs]() is: If there are `dead_1` dead features in layer `layer_1` and `dead_2` dead features in layer `layer_2`, we expect there to be `(d_sae * d_sae) - (d_sae - dead_1) * (d_sae - dead_2)` dead feature pairs between layers `layer_1` and `layer_2`.

![Plot](/artefacts/dead_features/res_jb_sae_dead_features_17.5M_0.0_0.png)

#### Dead feature pairs  

### `src`
This is the actual code library from which functions are imported into the experiment notebooks.