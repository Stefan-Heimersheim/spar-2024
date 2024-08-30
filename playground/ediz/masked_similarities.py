import numpy as np
import os
import einops


def load_data():
    # Path to the directory containing the .npz files
    directory = "../../artefacts/active_features/"

    # Load the mask .npz file
    mask_file_path = os.path.join(directory, "res_jb_sae_active_features_rel_0.0_100_last.npz")
    mask_data = np.load(mask_file_path)
    mask = mask_data['arr_0']

    # Load the data .npz file
    data_file_path = "np_arrays/res_jb_sae_feature_similarity_pearson_correlation_1M_0.0_0.1.npz"
    data = np.load(data_file_path)['arr_0']
    return mask, data

def generate_synthetic_data():
    # Parameters for synthetic data generation
    num_samples = 100   # number of mask slices
    num_features = 12   # number of features (first dimension of data)
    size = 1024         # reduced size for the 2D mask and 3D data array

    # Generate a random mask with True/False values
    mask = np.random.choice([True, False], size=(num_samples, num_features, size))

    # Generate random data with floating point numbers
    data = np.random.rand(num_features, size, size).astype(np.float32)

    return mask, data

def apply_mask_loop(mask: np.ndarray, data: np.ndarray, i: int) -> np.ndarray:
    """
    Original function using loops.
    """
    # Ensure i is within bounds
    if i < 0 or i >= mask.shape[0]:
        raise IndexError("Index i is out of bounds for the mask array.")

    # Get the mask slice for the given index i
    current_mask = mask[i]  # shape = (12, 24576)

    # Iterate over each dimension in the mask
    for x in range(current_mask.shape[0]):
        for y in range(current_mask.shape[1]):
            if not current_mask[x, y]:
                # Ensure indices are within bounds for the data array
                if x < data.shape[0] and y < data.shape[1]:
                    # Set data[x, y, :] and data[x, :, y] to 0
                    data[x, y, :] = 0
                    data[x, :, y] = 0

    return data

def apply_mask_optimized(mask: np.ndarray, data: np.ndarray, i: int) -> np.ndarray:
    """
    Optimized function using advanced indexing, matching the loop-based logic.
    """
    # Ensure i is within bounds
    if i < 0 or i >= mask.shape[0]:
        raise IndexError("Index i is out of bounds for the mask array.")
    
    # Get the mask slice for the given index i
    current_mask = mask[i]  # shape = (12, 1024)
    
    # Find the indices where the mask is False
    x_indices, y_indices = np.where(~current_mask)
    print(f"x_indices = {x_indices.shape}, y_indices = {y_indices.shape}, not {np.where(~current_mask)}")
    
    # Instead of zeroing out entire rows/columns, apply the changes selectively
    for x, y in zip(x_indices, y_indices):
        data[x, y, :] = 0
        data[x, :, y] = 0

    return data

def apply_mask_with_einops(mask: np.ndarray, data: np.ndarray, i: int) -> np.ndarray:
    """
    Optimized function using einops to broadcast the mask from (12, 24k) to (12, 24k, 24k).
    """
    # Ensure i is within bounds
    if i < 0 or i >= mask.shape[0]:
        raise IndexError("Index i is out of bounds for the mask array.")
    
    # Get the mask slice for the given index i
    current_mask = mask[i]  # shape = (12, 24k)

    # Use einops to repeat the mask across a new dimension
    expanded_mask_1 = einops.repeat(current_mask, 'x y -> x y z', z=data.shape[2])
    expanded_mask_2 = einops.repeat(current_mask, 'x y -> x z y', z=data.shape[2])

    # Apply the mask to the data
    data[~expanded_mask_1] = 0
    data[~expanded_mask_2] = 0

    return data

# Test function to compare both implementations
def test_apply_mask(mask, data):
    # Clone the data for comparison
    data_loop = data.copy()
    data_optimized = data.copy()

    # Choose a random index
    i = np.random.randint(0, mask.shape[0])
    print(f"Testing with index: {i}")

    # Apply both functions
    result_loop = apply_mask_loop(mask, data_loop, i)
    result_optimized = apply_mask_with_einops(mask, data_optimized, i)

    # Compare the results in detail
    if not np.array_equal(result_loop, result_optimized):
        print("Difference detected!")
        diff = np.where(result_loop != result_optimized)
        print("Indices of difference:", diff)
        print("Loop-based result at those indices:", result_loop[diff])
        print("Optimized result at those indices:", result_optimized[diff])
    else:
        print("Test passed! Both implementations produce the same result.")

# Generate synthetic data and run the test
mask, data = generate_synthetic_data()
test_apply_mask(mask=mask, data=data)
