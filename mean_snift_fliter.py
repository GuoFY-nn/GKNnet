import pickle
import numpy as np
from sklearn.cluster import MeanShift
from collections import Counter

def generate_unique_node_name(global_idx):
    return f"x{global_idx}"


pkl_filename = '/home/public_space/guofengyi/temp1/DRR095880/node/append_ref.pkl'


with open(pkl_filename, 'rb') as f:
    append_ref = pickle.load(f)


min_sv_length = 0


flattened_features = []
global_counter = 0
for sublist in append_ref:
    for item in sublist:
        svlength = float(item[1])

        if svlength >= min_sv_length:
            node_name = generate_unique_node_name(global_counter)
            features = item[1:]
            flattened_features.append((node_name, *features))
            global_counter += 1


chromosome_groups = {}
for item in flattened_features:
    chromosome = item[-1]
    if chromosome not in chromosome_groups:
        chromosome_groups[chromosome] = []
    chromosome_groups[chromosome].append(item)


total_clusters = 0

min_nodes_per_cluster = 0

output_bed_filename = 'temp1/DRR095880/cluster/filter.bed'
output_node_filename = 'temp1/DRR095880/cluster/filter_all.bed'
with open(output_bed_filename, 'w') as bed_file, open(output_node_filename, 'w') as node_file:

    for chromosome in sorted(chromosome_groups.keys()):
        nodes = chromosome_groups[chromosome]

        features = np.array([[float(node[1]), float(node[2]), float(node[3])] for node in nodes])

        if len(features) > 1:
            ms = MeanShift(bandwidth=100)
            ms.fit(features)
            labels = ms.labels_
        else:
            labels = [0]

        clusters = {}
        for label, node in zip(labels, nodes):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)


        node_positions = [(node[2], node[3]) for node in nodes]
        position_counter = Counter(node_positions)

        frequency_threshold = 0

        filtered_clusters = {
            label: [node for node in nodes if position_counter[(node[2], node[3])] >= frequency_threshold]
            for label, nodes in clusters.items() if len(nodes) >= min_nodes_per_cluster
        }

        total_clusters += len(filtered_clusters)

        for label, nodes in filtered_clusters.items():
            if len(nodes) > 0:
                nodes.sort(key=lambda x: x[2])

                midpoints = [(node[2] + node[3]) / 2 for node in nodes]
                average_midpoint = sum(midpoints) / len(midpoints)

                lengths = [node[3] - node[2] for node in nodes]
                average_length = sum(lengths) / len(lengths)

                chromStart = int(average_midpoint - average_length / 2)
                chromEnd = int(average_midpoint + average_length / 2)


                print(f"Cluster {label} on {chromosome}:")
                print(f"  Cluster Start: {chromStart}, Cluster End: {chromEnd}")
                for node_name, length, start, end, _ in nodes:
                    print(f'  Node: {node_name}, Length: {length}, Start: {start}, End: {end}')


                node_file.write(f"Cluster {label} on {chromosome}:  Cluster Start: {chromStart}, Cluster End: {chromEnd}\n")

                for node_name, length, start, end, _ in nodes:
                    node_file.write(f"    Node: {node_name}, Length: {length}, Start: {start}, End: {end}\n")


                bed_file.write(f"{chromosome}\t{chromStart}\t{chromEnd}\tDEL\n")


print(f"Total number of clusters: {total_clusters}")
print(f"Clusters saved to {output_bed_filename}")
