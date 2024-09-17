import os
import json

def generate_sample_list(folder_path, output_file='sample_list.json'):
    sample_files = []

    # Traverse the directory tree recursively
    for root, dirs, files in os.walk(folder_path):
        # Modify the dirs in-place to exclude directories starting with a dot
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.json'):
                # Append the relative path of the JSON file
                relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                sample_files.append(relative_path)

    # Create a dictionary structure as per the JSON file format
    sample_list = {"samples": sample_files}

    # Write the dictionary to the specified output file
    with open(output_file, 'w') as f:
        json.dump(sample_list, f, indent=4)

    print(f"Sample list generated at {output_file}")

if __name__ == "__main__":
    folder_path = "./graph_samples"
    generate_sample_list(folder_path)

