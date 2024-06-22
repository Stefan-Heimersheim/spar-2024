# Documentation of [`michael-0.py`](/playground/michael-0.py)
## Pearson Correlation between features in subsequent layers
### Idea
Get feature activations in subsequent layers, compute their Pearson correlation and see where this correlation exceeds a threshold

### Design choices / assumptions
- Use a 32-element batch with `seq_len=128` from the Pile as the dataset
- Only look at layers 6-9
- Only use first 100 features of each layer
- Set threshold to 0.2

### Results
Out of 10,000 feature pairs per pair of layers, 5-10 have a correlation of >= 0.2. Note: The columns are the 100 features of each layer under consideration (i.e., 6, 7, 8, 9).
![plot](/playground/michael-0-pearson-0.png)

## Co-occurrence between features in subsequent layers
### Idea
Get feature activations in subsequent layers, compute their co-occurrence and see where it exceeds a threshold

### Design choices / assumptions
- Only look at layers 6-9
- Only use first 100 features of each layer
- Set threshold to 0.2

### Results
Out of 10,000 feature pairs per layer pair, all have a co-occurrence of >= 0.2.
![plot](/playground/michael-0-cooccurrence-0.png)

The histogram (here between layers 6 and 7) shows the distribution of co-occurrences more clearly:
![plot](/playground/michael-0-cooccurrence-1.png)

## Co-occurrence between features in subsequent layers (2)
### Design choices / assumptions
- Only consider co-occurrences where not both activations are zero, i.e., compute `(#(both are > 0) / #(at least one is > 0))`

### Results
Out of 10,000 feature pairs per layer pair, 3-5 have a co-occurrence of >= 0.2.
![plot](/playground/michael-0-cooccurrence-2.png)