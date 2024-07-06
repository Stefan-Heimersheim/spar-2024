# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from src.similarity_helpers import load_correlation_data
# %%
# load jaccard results
folder = 'artefacts/jaccard'

file_basename = 'jaccard_2024-07-02-a_layer_X.npz'
layers = list(range(11))
filenames = [file_basename.replace('X', str(layer)) for layer in layers]
print(filenames)
# %%
def print_file_stats(file: str) -> float:
    data = np.load(file)['arr_0']
    non_zero_percentage = 100 * np.count_nonzero(data) / data.size
    print(f'File: {file}')
    print(f'Shape: {data.shape}')
    print(f'Size: {data.size}')
    print(f'Non-zero: {np.count_nonzero(data)}')
    print(f'% non-zero: {non_zero_percentage}')
    print()
    return non_zero_percentage
# %%
non_zero_percentages = []
for filename in filenames:
    non_zero_percentages.append(print_file_stats(os.path.join(folder, filename)))
# %%
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(layers, non_zero_percentages, marker='o', linestyle='-', color='b')
plt.title('Percentage of Non-Zero Elements in Jaccard Results by Layer Pair')
plt.xlabel('First layer in pair')
plt.ylabel('% Non-Zero Elements')
plt.grid(True)
plt.show()
# %%
from src.similarity_helpers import load_correlation_data
full_filenames = [os.path.join(folder, filename) for filename in filenames]
jaccard = load_correlation_data(full_filenames)
# %%
# %%
from src.similarity_helpers import save_compressed
save_compressed(jaccard, 'artefacts/jaccard/jaccard_2024-07-02-a.npz')
# %%
