import os
import shutil
from pathlib import Path
from typing import List, Set
import networkx as nx

from ediz_6 import read_partition_from_file

def copy_partition_files(partitions: List[Set[int]], source_dir: str, dest_dir: str, n: int, max_files: int) -> None:
    """
    Copies files from the source directory to the destination directory based on partition size.

    Args:
        partitions: The list of partitions from Louvain community detection.
        source_dir: The directory containing the original HTML files.
        dest_dir: The directory where the selected files will be copied.
        n: The number of nodes required in a partition to copy its file.
        max_files: The maximum number of files to copy.
    """
    # Ensure the destination directory exists
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    files_copied = 0

    for i, partition in enumerate(partitions):
        if len(partition) == n:
            source_file = Path(source_dir) / f"partition_{i + 1}.html"
            if source_file.exists():
                dest_file = Path(dest_dir) / source_file.name
                shutil.copyfile(source_file, dest_file)
                files_copied += 1

                if files_copied >= max_files:
                    print(f"Max file limit reached: {files_copied} files copied.")
                    break

    print(f"Total files copied: {files_copied}")

def process_directories(base_dir: str, partitions_dir:str, selection_root: str, nodes_required: List[int], max_files: int) -> None:
    """
    Iterates through all directories in the specified location, applying the copy_partition_files function.

    Args:
        base_dir: The base directory where all target directories are located.
        selection_root: The root directory where the selection of files will be stored.
        nodes_required: A list of node counts that will trigger the file copying process.
        max_files: The maximum number of files to copy.
    """
    # Ensure the selection root directory exists
    Path(selection_root).mkdir(parents=True, exist_ok=True)

    for directory in os.listdir(base_dir):
        dir_path = Path(base_dir) / directory
        if not dir_path.is_dir():
            print(f"Skipping non-directory: {directory}")
            continue

        partition_path = Path(partitions_dir) / f"{directory}.pkl"
        source_directory = dir_path / "plotly"
        destination_directory = Path(selection_root) / directory

        if not partition_path.exists():
            print(f"Partition file missing: {partition_path}")
            continue

        if not source_directory.exists():
            print(f"'plotly' directory missing in {dir_path}")
            continue

        try:
            partitions = read_partition_from_file(partition_path)
            for n in nodes_required:
                copy_partition_files(partitions, source_directory, destination_directory, n, max_files)
        except Exception as e:
            print(f"Error processing {directory}: {e}")

# Example usage:
if __name__ == "__main__":
    images_directory = "community_images"  # Replace with your actual base directory
    partitions_dir = "partitions"

    selection_root = "selection"  # Root directory for storing the selected files
    nodes_required = [4, 5, 6, 7, 8]  # Replace with the number of nodes required in a partition
    max_files_to_copy = 20  # Replace with the maximum number of files to copy

    process_directories(images_directory, partitions_dir, selection_root, nodes_required, max_files_to_copy)
