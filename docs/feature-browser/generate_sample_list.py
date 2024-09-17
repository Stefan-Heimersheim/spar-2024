import os
import json

def generate_sample_list(folder_path, output_file='sample_list.json'):
    # List all JSON files in the folder
    sample_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Create a dictionary structure as per the JSON file format
    sample_list = {"samples": sample_files}

    # Write the dictionary to a sample_list.json file
    with open(output_file, 'w') as f:
        json.dump(sample_list, f, indent=4)

    print(f"Sample list generated at {output_file}")

if __name__ == "__main__":
    folder_path = "./graph_samples"
    generate_sample_list(folder_path)
