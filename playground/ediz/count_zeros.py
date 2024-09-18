import numpy as np

def count_values_less_than_x(file_path, x):
    # Load the numpy array from the file
    array = np.load(file_path)['arr_0']
    
    # Count how many values are less than x
    count = np.sum(array < x)
    total_count = array.size
    
    return count, total_count


import numpy as np
import json

def convert_array_to_map(array, threshold):
    result = {}
    
    # Iterate through the 3D array
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                value = array[i, j, k]
                if value >= threshold:  # Keep only values >= threshold
                    result[(i, j, k)] = value
    
    return result

def save_to_json(data, file_path):
    # Save the dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    file_path = '../../artefacts/similarity_measures/pearson_correlation/res_jb_sae_feature_similarity_pearson_correlation_10M_0.1.npz'
    array = np.load(file_path)['arr_0']

    # Set a threshold value
    threshold = .1
    
    # Convert the array to a map and remove values below the threshold
    array_map = convert_array_to_map(array, threshold)
    
    # Save the map to a JSON file
    json_file_path = 'graph_jsons/pearson_10M_0.1.json'
    save_to_json(array_map, json_file_path)
    
    print(f"Data saved to {json_file_path}")
