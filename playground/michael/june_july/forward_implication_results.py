# %%
import numpy as np
import os
import matplotlib.pyplot as plt


# %%
# Load sufficiency results
folder = '../../artefacts/sufficiency'

file_basename = 'res_jb_sae_feature_correlation_sufficiency_X_Y_1M_0.1.npz'
layers = list(range(11))
filenames = [file_basename.replace('X', str(layer)).replace('Y', str(layer+1)) for layer in layers]
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
# Get file stats for all layer files
non_zero_percentages = []
for filename in filenames:
    non_zero_percentages.append(print_file_stats(os.path.join(folder, filename)))


# %%
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(layers, non_zero_percentages, marker='o', linestyle='-', color='b')
plt.title('Percentage of Non-Zero Elements in Sufficiency Results by Layer Pair')
plt.xlabel('First layer in pair')
plt.ylabel('% Non-Zero Elements')
plt.grid(True)
plt.show()

