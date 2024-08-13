import numpy as np
import sys

def generate_random_array_and_save(l: int, d: int, filename: str = "random_array.npz") -> None:
    """
    Generates a random numpy array of shape (l, d, d) with values between -1 and 1,
    and saves it to a .npz file.

    Args:
        l: Number of layers.
        d: Dimension of each layer.
        filename: The name of the file where the array will be saved.
    """
    # Generate a random numpy array of shape (l, d, d) with values between -1 and 1
    random_array = np.random.uniform(-1, 1, (l, d, d))
    
    # Save the array to a .npz file
    np.savez(filename, random_array)
    print(f"Array saved to {filename}")

if __name__ == "__main__":
    # Default values
    l = 11  # Number of layers
    d = 24  # Dimension of each layer
    file_name = 'test_graph.npz'

    # Check for command-line arguments
    if len(sys.argv) > 1:
        try:
            l = int(sys.argv[1])  # First argument is l
            d = int(sys.argv[2])  # Second argument is d
        except (ValueError, IndexError):
            print("Invalid input. Using default values for l and d.")

    generate_random_array_and_save(l, d, file_name)
