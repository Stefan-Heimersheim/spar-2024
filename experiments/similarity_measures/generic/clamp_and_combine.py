# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import einops
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from similarity_helpers import clamp_and_combine


# %%
print(torch.cuda.device_count())  # Should print 1
print(torch.cuda.current_device())  # Should print 0 since it's the first visible device
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# %%
measure_name = "mutual_information"

clamp_and_combine(
    measure_name, 
    base_folder='../../../artefacts/similarity_measures',
    clamping_threshold=0.1, 
    n_tokens='1M'
)