import pickle
import pysam
import torch_geometric
import re
import networkx as nx
import numpy as np
import torch
import os
from torch_geometric.data import HeteroData

def generate_edge_index_from_adj_matrix(adj_matrix):
    edge_index = np.array(adj_matrix.nonzero()).T
    weights = np.array(adj_matrix[edge_index[:, 0], edge_index[:, 1]])
    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(weights, dtype=torch.float32)

def generate_graphs(cigar_batch):
    all_graphs = []

    for cigar_string, ref_start, chromosome in cigar_batch:
        cigar_tuples = re.findall(r'(\d+)([A-Z])', cigar_string)
        operation_mapping = {'M': 0, 'I': 1, 'S': 3, 'D': 2, 'H': 3}
        cigar_tuples_converted = [(int(number), operation_mapping[letter]) for number, letter in cigar_tuples]

        G = nx.Graph()
        ref_nodes = []
        read_nodes = []
        pos = ref_start
        ref_pos = 0
        read_pos = 0

        for length, op in cigar_tuples_converted:
            features = np.zeros(6)
            features[op] = length
            features[4] = pos
            features[5] = pos + length - 1
            if length < 50 and op in (2, 1, 3):
                pos += length
                continue
            if op == 3 and length >= 50:
                label = 3
                read_node = f"read_{read_pos}"
                G.add_node(read_node, features=features, label=label, chromosome=chromosome)
                read_nodes.append(read_node)
                pos += length
                read_pos += 1
            elif op == 2 and length >= 50:
                label = 2
                ref_node = f"ref_{ref_pos}"
                G.add_node(ref_node, features=features, label=label, chromosome=chromosome)
                ref_nodes.append(ref_node)
                ref_pos += 1
                pos += length
            elif op == 0:
                label = 0
                ref_node = f"ref_{ref_pos}"
                read_node = f"read_{read_pos}"
                G.add_node(ref_node, features=features, label=label, chromosome=chromosome)
                G.add_node(read_node, features=features, label=label, chromosome=chromosome)
                G.add_edge(ref_node, read_node)
                ref_nodes.append(ref_node)
                read_nodes.append(read_node)
                ref_pos += 1
                read_pos += 1
                pos += length
            elif op == 1 and length >= 50:
                label = 1
                read_node = f"read_{read_pos}"
                G.add_node(read_node, features=features, label=label, chromosome=chromosome)
                read_nodes.append(read_node)
                read_pos += 1
                pos += length

        for i in range(len(ref_nodes) - 1):
            G.add_edge(ref_nodes[i], ref_nodes[i + 1])
        for i in range(len(read_nodes) - 1):
            G.add_edge(read_nodes[i], read_nodes[i + 1])

        A1 = nx.adjacency_matrix(G, nodelist=read_nodes + ref_nodes)
        A2 = nx.adjacency_matrix(G, nodelist=ref_nodes)
        A3 = nx.adjacency_matrix(G, nodelist=read_nodes)

        edge_index_A1, weight_A1 = generate_edge_index_from_adj_matrix(A1)
        edge_index_A2, weight_A2 = generate_edge_index_from_adj_matrix(A2)
        edge_index_A3, weight_A3 = generate_edge_index_from_adj_matrix(A3)

        features = np.array([G.nodes[node]['features'] for node in G.nodes])
        x = torch.tensor(features, dtype=torch.float32)
        node_labels = np.array([G.nodes[node]['label'] for node in G.nodes])
        y = torch.tensor(node_labels, dtype=torch.float32)


        data = HeteroData()


        data['read'].x = torch.tensor([G.nodes[n]['features'] for n in read_nodes], dtype=torch.float32)
        data['ref'].x = torch.tensor([G.nodes[n]['features'] for n in ref_nodes], dtype=torch.float32)


        data['read', 'aligned_to', 'ref'].edge_index = edge_index_A1
        data['read', 'aligned_to', 'ref'].edge_attr = weight_A1

        data['ref', 'ref_adjacent', 'ref'].edge_index = edge_index_A2
        data['ref', 'ref_adjacent', 'ref'].edge_attr = weight_A2

        data['read', 'read_adjacent', 'read'].edge_index = edge_index_A3
        data['read', 'read_adjacent', 'read'].edge_attr = weight_A3


        data['read'].y = torch.tensor([G.nodes[n]['label'] for n in read_nodes], dtype=torch.float32)
        data['ref'].y = torch.tensor([G.nodes[n]['label'] for n in ref_nodes], dtype=torch.float32)

        all_graphs.append(data)

    return all_graphs

def parse_first_bam_with_cigar(bam_file):
    cigar_info = []
    cigar_list = []

    bam = pysam.AlignmentFile(bam_file, "rb")


    for i, read in enumerate(bam):

        if read.cigarstring:

            read_name = read.query_name
            reference_name = bam.get_reference_name(read.reference_id)
            reference_start = read.reference_start
            reference_end = read.reference_end
            cigar_string = read.cigarstring
            sequence_length = read.query_length
            sequence_start = read.reference_start
            chromosome = read.reference_name


            cigar_info.append({
                "read_name": read_name,
                "chromosome": chromosome,
                "reference_start": reference_start,
                "sequence_start": sequence_start,
                "sequence_length": sequence_length,
                "reference_end": reference_end,
                "cigar_string": cigar_string
            })
            cigar_list.append((cigar_string, reference_start, chromosome))


    bam.close()
    return cigar_info, cigar_list

def save_dataset(graphs, counter, save_path):
    filename = os.path.join(save_path, f"dataset_{counter}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(graphs, f)

    print(f"Dataset {counter} saved successfully to {filename}")

def generate_graphs_in_batches(bam_file, batch_size, dataset_size):
    total_graphs = []
    _, all_cigar_strings = parse_first_bam_with_cigar(bam_file)
    save_counter = 1

    save_path = "/temp1/ERR8562466/graphdata"
    for i in range(0, len(all_cigar_strings), batch_size):

        cigar_batch = all_cigar_strings[i:i + batch_size]
        batch_graphs = generate_graphs(cigar_batch)
        total_graphs.extend(batch_graphs)

        if len(total_graphs) >= 5000:
            save_dataset(total_graphs, save_counter, save_path)
            total_graphs = []
            save_counter += 1


    if total_graphs:
        save_dataset(total_graphs, save_counter, save_path)




def main():
    bam_file = "temp1/ERR8562466/aln_sort_filter.bam"
    graphs = generate_graphs_in_batches(bam_file, batch_size, dataset_size)



if __name__ == "__main__":
    main()
