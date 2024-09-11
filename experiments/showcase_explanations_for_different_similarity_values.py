# %%
import os
import numpy as np
import sys
import pickle
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from similarity_helpers import get_filename, load_similarity_data


# %%
# Load similarity matrices
sae_name = 'res_jb_sae'
n_layers = 12

measure_name = "pearson_correlation"
activation_threshold = None

folder = f'../../artefacts/similarity_measures/{measure_name}/.unclamped'
similarities = load_similarity_data([f'{folder}/{get_filename(measure_name, "feature_similarity", activation_threshold, None, n_tokens="10M", first_layer=layer, sae_name=sae_name)}.npz' for layer in range(n_layers - 1)])


# %%
def approximate_sample_indices_large_3d(arr, sample_values, sample_size=1000000):
    print('Sampling from the similarity matrix...')
    # Get the shape of the array
    shape = arr.shape
    total_elements = np.prod(shape)
    
    # Generate random indices for sampling
    random_indices = np.random.choice(total_elements, size=sample_size, replace=True)
    
    # Get the samples
    samples = arr.ravel()[random_indices]
    
    print('Sorting the samples...')
    # Sort the samples and get the sorted indices
    np.nan_to_num(samples, copy=False)
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    
    
    print('Finding closest entries...')
    # Find the approximate indices
    approx_indices = []
    for value in tqdm(sample_values):
        if value == np.min(arr):
            idx = np.argmin(arr)
        elif value == np.max(arr):
            idx = np.argmax(arr)
        else:
            # Find the closest value in our sorted samples
            closest_idx = np.searchsorted(sorted_samples, value)
            if closest_idx == len(sorted_samples) or (closest_idx > 0 and 
               value - sorted_samples[closest_idx-1] < sorted_samples[closest_idx] - value):
                closest_idx -= 1
            
            # Get the original index in the large array
            idx = random_indices[sorted_indices[closest_idx]]
        
        # Convert flat index to 3D coordinates
        approx_indices.append(np.unravel_index(idx, shape))
    
    return np.array(approx_indices)

sample_values = [0.0, 0.2, 0.5, 0.9, 0.95, 0.98]

min_index = np.array([np.unravel_index(np.nanargmin(similarities), similarities.shape)])
other_indices = approximate_sample_indices_large_3d(similarities, sample_values)


# %%
result_indices = np.concatenate([min_index, other_indices])

print("Indices of samples (z, y, x):", result_indices)
print("Values at these indices:", similarities[tuple(result_indices.T)])


# %%
with open('../../artefacts/explanations/res_jb_sae_explanations.pkl', 'rb') as f:
    explanations = pickle.load(f)

for l, f1, f2 in result_indices:
    print(f'{similarities[l, f1, f2]:.2f} & \\verb|{l}/{f1}| ({explanations[l][f1]}) & \\verb|{l+1}/{f2}| ({explanations[l+1][f2]}) \\\\ \hline')
