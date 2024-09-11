# %%
import re
from sae_lens import SAE
from tqdm import tqdm
import torch
import torch as t
import typing
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import List, Dict
import torch as t
import torch
import numpy as np
import typing
import einops
from sae_lens import SAE
from tqdm import tqdm
from functools import partial
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from src import D_SAE
# %%
# %%
pearson_corr_filename = f"artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz"
with open(pearson_corr_filename, 'rb') as data:
    pearson_corr = np.load(data)['arr_0']

# %%
sufficiency_filename = f"artefacts/similarity_measures/sufficiency_relative_activation/res_jb_sae_feature_similarity_sufficiency_relative_activation_10M_0.2_0.1.npz"
with open(sufficiency_filename, 'rb') as data:
    sufficiency = np.load(data)['arr_0']

# %%
necessity_filename = f"artefacts/similarity_measures/necessity_relative_activation/res_jb_sae_feature_similarity_necessity_relative_activation_10M_0.2_0.1.npz"
with open(necessity_filename, 'rb') as data:
    necessity = np.load(data)['arr_0']

# %%
feature_pairs = [
    ((5, 16673), (6, 6645)),
    ((5, 16673), (6, 2653)),
    ((6, 6645), (7, 5871)),
    ((6, 2653), (7, 5871))
]

for ((layer1, feat1), (layer2, feat2)) in feature_pairs:
    print(f"Layer {layer1}, Feature {feat1} vs Layer {layer2}, Feature {feat2}")
    print(f"Pearson Correlation: {pearson_corr[layer1, feat1, feat2]}")
    print(f"Sufficiency Relative Activation: {sufficiency_filename[layer1, feat1, feat2]}")
# %%
