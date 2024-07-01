# Helper function for existing correlation matrices
#
# 1. Load correlation matrix files (one per layer pair)
# 2. Stack layers into one big matrix
# 3. Cut off values below given threshold
# 4. Save compressed version as one file

# %%
from typing import List
import numpy as np


# %%
def load_correlation_data(files: List[str]):
    if len(files) == 1:  # Single file
        data = np.load(files[0])['arr_0']
        
        return data
    else:  # Multiple files
        data = [np.load(file)['arr_0'] for file in files]

        return np.stack(data)

def clamp_low_values(tensor, threshold):
    tensor[tensor < threshold] = 0


def save_compressed(tensor, filename):
    np.savez_compressed(filename, tensor.detach().cpu().numpy())