import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
import pickle
from model_kan import GcnSVModel
import numpy as np


all_conf_matrix = np.zeros((2, 2))
all_total_true_positives = 0
all_total_actual_positives = 0
all_total_predicted_positives = 0
dataset_numbers = range(1, 2)
for weight in []:
    for m in dataset_numbers:
        # 假设您的.pkl文件路径如下
        pkl_filename = 'temp1/DRR095880/graphdata/dataset_{}.pkl'.format(m)
        # 使用pickle加载.pkl文件
        with open(pkl_filename, 'rb') as f:
            loaded_graphs = pickle.load(f)
            limited_graphs = loaded_graphs[:500]


        data_loader = DataLoader(loaded_graphs, batch_size=batch_size, shuffle=True)

        torch.manual_seed(1234)

        total_samples = 0
        total_correct = 0
        total_TP = 0
        total_FN = 0
        total_FP = 0
        all_true_y = []
        all_predicted = []

        total_true_positives = 0
        total_actual_positives = 0
        total_predicted_positives = 0

        model_path = 'kan/model/kan+fc/best_model_{}.pth'.format(weight)
        test_model_w = torch.load(model_path)
        test_model = GcnSVModel(input_features=6, hidden_features=12, output_classes=2)
        test_model.load_state_dict(test_model_w)
        test_model.eval()
        with torch.no_grad():
            for batch in data_loader:

                outputs = test_model(batch)

                #predicted = (outputs > 0.5).long()
                predicted = torch.argmax(outputs, dim=1)

                true_y = batch.y.view(-1).long()

                all_true_y.extend(true_y.cpu().numpy())
                all_predicted.extend(predicted.view(-1).cpu().numpy())

                batch_conf_matrix = confusion_matrix(true_y.cpu().numpy(), predicted.view(-1).cpu().numpy())
                # print(f'Confusion Matrix:')
                # print(batch_conf_matrix)

                all_conf_matrix += batch_conf_matrix

                correct_rows = torch.eq(predicted.view(-1), true_y).sum().item()

                total_correct += correct_rows
                total_samples += true_y.size(0)

                if batch_conf_matrix.shape[0] > 1 and batch_conf_matrix.shape[1] > 1:
                    tp = batch_conf_matrix[1, 1]
                    fn = batch_conf_matrix[1, 0]
                    batch_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    #print(f'Recall of the model on current batch (label 1): {batch_recall * 100:.2f}%')

                    fp = batch_conf_matrix[0, 1]  # False Positives
                    batch_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    #print(f'Precision of the model on current batch (label 1): {batch_precision * 100:.2f}%')

                    total_true_positives += tp
                    all_total_true_positives += tp
                    total_actual_positives += (tp + fn)
                    all_total_actual_positives += (tp + fn)
                    total_predicted_positives += (tp + fp)
                    all_total_predicted_positives += (tp + fp)


        accuracy = total_correct / total_samples
        print(f'Accuracy: {accuracy:.4f}%')


        conf_matrix = confusion_matrix(all_true_y, all_predicted)
        print('all_Confusion Matrix dataset_{} :'.format(m))
        print(conf_matrix)


        overall_recall = total_true_positives / total_actual_positives if total_actual_positives > 0 else 0
        print(f'Overall Recall (label 1) dataset_{m} : {overall_recall * 100:.2f}%')


        overall_precision = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0
        print(f'Overall Precision (label 1) dataset_{m} : {overall_precision * 100:.2f}%')


    all_overall_recall = all_total_true_positives / all_total_actual_positives if all_total_actual_positives > 0 else 0
    print(f'Overall Recall (label 1) dataset_{weight} : {all_overall_recall * 100:.2f}%')

    all_overall_precision = all_total_true_positives / all_total_predicted_positives if all_total_predicted_positives > 0 else 0
    print(f'Overall Precision (label 1) dataset_{weight} : {all_overall_precision * 100:.2f}%')

