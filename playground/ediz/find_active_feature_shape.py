import numpy as np
import os

# Path to the directory containing the .npz files
directory = "../../artefacts/active_features/"

# List all files in the directory
files = os.listdir(directory)

# Iterate over each file in the directory
for file in files:
    # Construct the full path to the file
    file_path = os.path.join(directory, file)
    
    if file.endswith('.npz'):
        print(f"Loading {file}...")
        # Load the .npz file
        data = np.load(file_path)

        # Iterate over each array in the .npz file
        for key in data.files:
            print(f"Array '{key}' shape: {data[key].shape}")
        
        # Close the file to free up memory
        data.close()
        print(f"Finished processing {file}.\n")
