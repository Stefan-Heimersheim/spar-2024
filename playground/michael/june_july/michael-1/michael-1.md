# Documentation of [`michael-1.py`](/playground/michael-1.py)
## Cosine similarity between features in subsequent layers
### Idea
Get feature directions in subsequent layers, compute their cosine similarity and see where this correlation exceeds a threshold

#### Reasoning
Since the residual connections transfer the residual stream from one layer to the next without any transformation, it seems plausible that SAEs in subsequent layers have features which activate for similar directions of the residual stream. Such similarities can be captured by the cosine similarity of their decoder weights.

### Design choices / assumptions
- Only look at layers 6-9
- Only use first 100 features of each layer
- Set threshold to 0.5

### Results
There seems to be at least some overlap between SAE feature directions from different layers:
![plot](/playground/michael-1-sae_feature_pca-0.png)

However, the similarity analysis does not (yet) provide meaningful results:
> Top feature pairs between layers 6 and 7:
> 
> Similarity of 0.5547 found between SAE features 6_50 (references to firearms, specifically rifles) and 7_74 (words related to making assumptions or assertions).
> 
> Similarity of 0.5213 found between SAE features 6_71 (mentions of the time "midnight") and 7_25 (phrases indicating a high level of excitement or anticipation).
> 
> Similarity of 0.4429 found between SAE features 6_5 (mentions of individuals working or collaborating on various projects) and 7_22 (None).
> 
> Similarity of 0.4260 found between SAE features 6_68 (None) and 7_14 (acronyms in the form of three capitalized letters).
> 
> Similarity of 0.4088 found between SAE features 6_50 (references to firearms, specifically rifles) and 7_70 ( discussions or mentions of interest rates).
> Similarity of 0.3760 found between SAE features 6_63 (words related to pepper in various contexts) and 7_78 (None).
> 
> Similarity of 0.3657 found between SAE features 6_5 (mentions of individuals working or collaborating on various projects) and 7_77 (words related to reclamation or recovery).
> 
> Similarity of 0.3494 found between SAE features 6_9 (numerical data and scientific measurements) and 7_65 (None).
> 
> Similarity of 0.3452 found between SAE features 6_7 (references to the word "Char" followed by a digit) and 7_91 (references to events or activities related to hosting).
> 
> Similarity of 0.3415 found between SAE features 6_15 (terms related to achieving peak performance or reaching maximum levels) and 7_81 (words related to synthetic materials or compounds).

> Top feature pairs between layers 7 and 8:
> 
> Similarity of 0.9030 found between SAE features 7_75 (words related to Irish culture) and 8_45 (mentions of the state of Wyoming).
> 
> Similarity of 0.4992 found between SAE features 7_97 (None) and 8_91 (the word "imagine" in various contexts).
> 
> Similarity of 0.4556 found between SAE features 7_38 (keywords related to online forums) and 8_6 (references to a specific entity named "Kal").
> 
> Similarity of 0.4190 found between SAE features 7_9 (None) and 8_42 (It seems there is no clear pattern of activation for Neuron 4 as it does not activate for any of the provided tokens. Without any non-zero activations to analyze, we cannot determine what this neuron is looking for).
> 
> Similarity of 0.4144 found between SAE features 7_17 (terms related to intelligence agencies and activities) and 8_33 (It seems like there is an issue with Neuron 4's activations, as all activation values are zero, which indicates that it does not activate for any part of the text provided. Without non-zero activation values, it's not possible to determine what the neuron is looking for).
> 
> Similarity of 0.3901 found between SAE features 7_8 (questions and inquiries) and 8_75 (words related to revisiting, reappearance, or resurfacing).
> 
> Similarity of 0.3723 found between SAE features 7_8 (questions and inquiries) and 8_25 (terms related to inheritance and succession).
> 
> Similarity of 0.3640 found between SAE features 7_92 (specific references to weeks in a sports context) and 8_25 (terms related to inheritance and succession).
> 
> Similarity of 0.3595 found between SAE features 7_17 (terms related to intelligence agencies and activities) and 8_5 (legal terms and procedures).
> 
> Similarity of 0.3281 found between SAE features 7_56 (None) and 8_13 (words related to physical tools or actions involving force).