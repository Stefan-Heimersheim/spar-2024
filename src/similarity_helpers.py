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

def clamp_low_values(arr, threshold):
    arr[np.abs(arr) < threshold] = 0
    return arr


def save_compressed(arr, filename):
    np.savez_compressed(filename, arr)


def get_n_token_description(n_tokens: int) -> str:
    def is_power_of_2(x):
        return (x != 0) and (x & (x - 1)) == 0

    if is_power_of_2(n_tokens):
        if n_tokens >= 2**30:
            return f"{n_tokens // 2**30}G"
        elif n_tokens >= 2**20:
            return f"{n_tokens // 2**20}M"
        elif n_tokens >= 2**10:
            return f"{n_tokens // 2**10}k"
        else:
            return str(n_tokens)
    else:
        if n_tokens >= 1_000_000_000:
            return f"{n_tokens / 1_000_000_000:.1f}B".rstrip('0').rstrip('.')
        elif n_tokens >= 1_000_000:
            return f"{n_tokens / 1_000_000:.1f}M".rstrip('0').rstrip('.')
        elif n_tokens >= 1_000:
            return f"{n_tokens / 1_000:.1f}k".rstrip('0').rstrip('.')
        else:
            return str(n_tokens)


def get_filename(measure_name: str, activation_threshold: float, clamping_threshold: float, n_tokens: int | str, first_layer: int = None, sae_name: str = 'res_jb_sae') -> str:
    if type(n_tokens) is int:
        n_tokens = get_n_token_description(n_tokens)

    filename = f'{sae_name}_feature_similarity_{measure_name}_{n_tokens}_{activation_threshold:.1f}_{clamping_threshold:.1f}'

    if first_layer is not None:
            filename += f'_{first_layer}'

    return filename
