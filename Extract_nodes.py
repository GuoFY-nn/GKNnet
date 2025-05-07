import torch
from torch_geometric.loader import DataLoader
import pickle
from model_kan import GcnSVModel
import numpy as np


def generate_unique_node_name(global_idx):
    return f"x{global_idx}"


dataset_numbers = range(1, 25)
append_ref = []
all_append_ref = []
global_counter = 0

test_model_w = torch.load('')
test_model = GKNnet(input_features=6, hidden_features=12, output_classes=2)
test_model.load_state_dict(test_model_w)
test_model.eval()

for m in dataset_numbers:

    pkl_filename = 'temp1/DRR095880/graphdata/dataset_{}.pkl'.format(m)


    with open(pkl_filename, 'rb') as f:
        loaded_graphs = pickle.load(f)
        limited_graphs = loaded_graphs[:10]


    data_loader = DataLoader(loaded_graphs, batch_size=batch_size, shuffle=True)


    with torch.no_grad():
        for batch in data_loader:

            temp_append_ref = []

            outputs = test_model(batch)
            node_info = batch.G_nodes

            predicted = torch.argmax(outputs, dim=1)
            temp_key = []
            for key, value in node_info.items():
                temp_key.append(key)
            for i in range(predicted.size(0)):
                if predicted[i].tolist() == 1:
                    if 'ref' in temp_key[i]:
                        value = node_info[temp_key[i]]
                        features_array = value['features'][0]
                        chromosome = value['chromosome'][0]
                        first_four_elements = features_array[:4]
                        max_value = np.max(first_four_elements)
                        temp_append_ref.append(
                            (temp_key[i], max_value, features_array[4], features_array[5], chromosome))
            if temp_append_ref:
                append_ref.append(temp_append_ref)
                all_append_ref.append(temp_append_ref)


    if (m - dataset_numbers.start + 1) % 10 == 0:
        output_filename = f'temp1/DRR095880/node/append_ref_{m}.pkl'
        with open(output_filename, 'wb') as f:
            pickle.dump(append_ref, f)
        print(f"append_ref 已成功保存到 {output_filename}")
        append_ref = []


if append_ref:
    output_filename = f'temp1/DRR095880/node/append_ref_final.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(append_ref, f)
    print(f"append_ref 已成功保存到 {output_filename}")
output_filename = f'temp1/DRR095880/node/append_ref.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(all_append_ref, f)
print(f"append_ref 已成功保存到 {output_filename}")
