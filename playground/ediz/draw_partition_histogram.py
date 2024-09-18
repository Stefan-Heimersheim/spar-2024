from src.partition_utils import read_partition_from_file, plot_partition_histogram

measures = ["jaccard_similarity", "mutual_information", "necessity","pearson","sufficiency"]
method = "louvain"
threshold = 0.1


for measure in measures:
    root_file = f"{measure}_{method}_threshold_{str(threshold)}_plotly"
    partition_file = f"partitions/{root_file}.pkl"
    histogram_file = f"histograms/{root_file}_log.png"
    partition = read_partition_from_file(partition_file)

    plot_partition_histogram(partition, histogram_file, log_y=True)
