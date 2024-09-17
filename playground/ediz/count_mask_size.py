import numpy as np
print(np.__version__)

# Load the .npz file
npz_file_path = '../../artefacts/active_features/res_jb_sae_active_features_rel_0.0_100_last.npz'  # Replace with the path to your .npz file
data = np.load(npz_file_path)

# Assuming you know the key to access the array, otherwise list all keys
print("Available keys in the .npz file:", data.files)

# Replace 'array_key' with the correct key that corresponds to the array
array_key = 'arr_0'  # Replace with the actual key
array = data[array_key][0]

# Ensure the array is of shape (a, b)
#if array.ndim != 3:
#    raise ValueError(f"The array does not have the expected shape of (a, b). Found shape: {array.shape}")

# Count the number of 'True' values in the array
true_count = np.sum(array == True)  # Alternatively, np.sum(array) also works if it's a boolean array
false_count = np.sum(array == False)
print(f"Number of 'True' values in the array: {true_count}")
print(f"Number of 'False' values in the array: {false_count}")
