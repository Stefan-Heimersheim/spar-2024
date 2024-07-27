# Helper functions for existing similarity matrices

from typing import List
import numpy as np


def load_similarity_data(files: List[str]):
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

    filename = f'{sae_name}_feature_similarity_{measure_name}_{n_tokens}_{activation_threshold:.1f}'

    if clamping_threshold is not None:
        filename += f'_{clamping_threshold:.1f}'

    if first_layer is not None:
        filename += f'_{first_layer}'

    return filename


def clamp_and_combine(measure_name: str, 
                      clamping_threshold: float, 
                      n_tokens: int | str, 
                      base_folder: str = '../../artefacts/similarity_measures', 
                      activation_threshold: float = 0.0,
                      n_layers: int = 12,
                      sae_name: str = 'res_jb_sae'
                      ) -> None:
    folder = f'{base_folder}/{measure_name}/.unclamped'
    files = [f'{folder}/{get_filename(measure_name, activation_threshold, clamping_threshold=None, n_tokens=n_tokens, first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)]
    
    matrix = load_similarity_data(files)
    matrix = np.nan_to_num(matrix)

    clamp_low_values(matrix, clamping_threshold)

    save_compressed(matrix, f'{base_folder}/{measure_name}/{get_filename(measure_name, activation_threshold, clamping_threshold, n_tokens, sae_name=sae_name)}')
