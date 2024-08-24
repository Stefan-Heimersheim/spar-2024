# %%
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
import random
import gc
import os
import pickle
import leidenalg as la
import igraph as ig
from ediz_6 import *

# Path to the partition file
partition_file = 'partitions/pearson_louvain.pkl'  # Replace with your actual file path

# Read the partition from the file
partition = read_partition_from_file(partition_file)

# Calculate the size of each partition (number of nodes)
partition_sizes = [len(community) for community in partition]

# Create a histogram of the partition sizes
plt.figure(figsize=(10, 6))
plt.hist(partition_sizes, bins=30, log=True, edgecolor='black')  # Log scale on the y-axis
plt.title('Histogram of Partition Sizes')
plt.xlabel('Number of Nodes in Partition')
plt.ylabel('Count (Log Scale)')

# Save the histogram to a file
histogram_file = 'histograms/pearson_louvain_partition_histogram.png'  # Replace with your desired file name
plt.savefig(histogram_file)
plt.close()

print(f"Histogram saved to {histogram_file}")
