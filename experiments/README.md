# Experiments
## Mapping of experiments and sections of the paper
All experiments in this folder correspond to specific results and/or plots in the paper. Below you can find the mapping, listing which code files produce which result.

### 3 Results
TODO: Figure 2
TODO: Table 1

#### 3.1 Features being "passed through" multiple layers
Figure 3 is produced by running `experiments/pass_through_analysis.py`.

#### 3.2 Logic Gates
Table 2 is produced by running `experiments/logic_gates.py`.

#### 3.3 Did features disappear, or did they lack representation in the next layer's SAE?
In Figure 4, the right charts were created by running `experiments/projected_errs/collect_per_token_projected_errs_and_acts.py`

#### 3.4 Community-detection finds semantically meaningful subgraphs
Figure 5 was created by going through all the communities in the [feature browser](https://stefanhex.com/spar-2024/feature-browser/)

### Appendix
#### B Distribution of maximum activation values per feature
Figure 7 is produced by running `experiments/max_activation_analysis.py`, and the raw data can be found in `artefacts/max_activation_analysis/`.

#### C Feature Activation Sampling
TODO: Figure 8 is produced by running 

Figure 9 is produced by running `experiments/compare_10_and_100M_tokens.py`.

#### D Distribution of similarity values
TODO: Figure 10 is produced by running

#### E Similarity measures
Figure 11 is produced by running `experiments/compare_10_and_100M_tokens.py`.

#### F Explanation pairs for different similarity values
Tables 3 to 6 are produced running `experiments/explanations_for_different_similarities.py`.

#### G Pass-through features at different thresholds
TODO: Figure 12 is produced by running 

#### I Relationship between ablation effect and similarity measure
Figure 13 was produced by running `experiments/ablations/sample_pairs.py` to collect indexes of features in adjacent layer pairs
Followed by running `experiments/ablations/ablate_sampled_pairs.py` to ablate the "previous layer" feature of each pair
and then finally through `experiments/ablations/analyze_ablations.py`

## Unordered list
- [DONE] Sankey diagram: 
    - `experiments/pass_through_analysis.py`
    - `artefacts/pass_through_analysis/`
- [DONE] Distribution of maximum SAE activations:
    - `experiments/max_activation_analysis.py`
    - `artefacts/max_activation_analysis/`
- [TODO] Distribution of co-activations:
    - `experiments/co_activation_analysis.py`
    - `artefacts/co_activation_analysis/`
- [TODO] Distribution of similarity values:
    - `experiments/activation_histograms/activation_histograms.py`
    - `artefacts/`
- [DONE] Boolean Relationships
    - `experiments/boolean_relationships.py`
    - (no artefacts, only terminal output)
- [DONE] Comparison of Pearson correlation on 10M and 100M tokens
    - `experiments/compare_10_and_100M_tokens.py`
    - (no artefacts, images are terminal output)
- [DONE] Comparison of centered (i.e., Pearson correlation) and uncentered cosine similarity
    - `experiments/compare_centered_and_uncentered_cosine.py`
    - (no artefacts, images are terminal output)
- [DONE] Explanation pairs for different similarity values
    - `experiments/compare_10_and_100M_tokens.py`
    - (no artefacts, tables are terminal output)