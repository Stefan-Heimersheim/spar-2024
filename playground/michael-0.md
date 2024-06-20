# Michael's notes
## Brainstorming
### Conjectures
- Features found in early layers are simpler and more syntax-related than features in later layers
- Features in later layers are combinations of earlier features, enriched with the "learned knowledge" of the model
- Features in different SAEs change gradually since the residual connection between layers transfers activations directly from one layer to the next
  - Could it be that a feature is systematically erased by an attention head whenever it is not represented in the next layer? With such a mechanism, the model could "clean up" unused features to free space for other features
  - Can we see this by looking for eraser neurons? Or do individual neurons both erase one feature and immediately replace it with another one?


### Ideas
#### Building and validating a causal graph
- [DONE] Use arbitrary input and do a statistical correlation analysis between features in different layers
- Build a tentative causal graph from the correlations
- Selectively patch activations so that features are turned on/off
- Derive the strength of causal connections from observing what changes at later layers

#### Connecting features with meaning
- Create a dataset of well-defined concepts that we expect to see represented as features
- Trace activations through the layer SAEs to see how these concepts are used and changed