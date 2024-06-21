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
- [DONE] Build a tentative causal graph from the correlations
- Selectively patch activations so that features are turned on/off
- Derive the strength of causal connections from observing what changes at later layers

#### Connecting features with meaning
- Create a dataset of well-defined concepts that we expect to see represented as features
- Trace activations through the layer SAEs to see how these concepts are used and changed

## Experiments
### Pearson Correlation between features in subsequent layers
#### Idea
Get feature activations in subsequent layers, compute their Pearson correlation and see where this correlation exceeds a threshold

#### Design choices / assumptions
- Use a 32-element batch with `seq_len=128` from the Pile as the dataset
- Only look at layers 6-9
- Only use first 100 features of each layer
- Set threshold to 0.2

#### Results
Out of 10,000 feature pairs per pair of layers, 5-10 have a correlation of >= 0.2. Note: The columns are the 100 features of each layer under consideration (i.e., 6, 7, 8, 9).
![plot](/playground/michael-0-pearson-0.png)

### Co-occurrence between features in subsequent layers
#### Idea
Get feature activations in subsequent layers, compute their co-occurrence and see where it exceeds a threshold

#### Design choices / assumptions
- Only look at layers 6-9
- Only use first 100 features of each layer
- Set threshold to 0.2

#### Results
Out of 10,000 feature pairs per layer pair, all have a co-occurrence of >= 0.2.
![plot](/playground/michael-0-cooccurrence-0.png)

The histogram (here between layers 6 and 7) shows the distribution of co-occurrences more clearly:
![plot](/playground/michael-0-cooccurrence-1.png)

### Co-occurrence between features in subsequent layers (2)
#### Design choices / assumptions
- Only consider co-occurrences where not both activations are zero, i.e., compute `(#(both are > 0) / #(at least one is > 0))`

#### Results
Out of 10,000 feature pairs per layer pair, 3-5 have a co-occurrence of >= 0.2.
![plot](/playground/michael-0-cooccurrence-2.png)