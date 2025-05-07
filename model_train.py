import torch
import pickle
import time
import os
from torch_geometric.loader import DataLoader
from model_kan import GKNnet
import torch.nn as nn
from sklearn.metrics import confusion_matrix

epochs = 50
dataset_numbers = range(1, 9)
dataset_numbers1 = range(9, 16)


torch.manual_seed(1234)


patience = 10
best_precision = 0
counter = 0

for weight in []:
    all_epoch_losses = []
    prec = 0


    model_path = f'temp1/ERR8562466/model/best_model_{weight}.pth'
    model = GKNnet(input_features=6, hidden_features=12, output_classes=2)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1-weight, weight]))

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs} 开始训练...")


        epoch_loss_total = 0


        for m in dataset_numbers:
            pkl_filename = f'/home/public_space/guofengyi/temp1/ERR8562466/graphdata/dataset_{m}.pkl'

            with open(pkl_filename, 'rb') as f:
                loaded_graphs = pickle.load(f)


            data_loader = DataLoader(loaded_graphs, batch_size=batch_size, shuffle=True)
            total_loss = 0

            for batch in data_loader:
                optimizer.zero_grad()
                out = model(batch)
                true_y = batch.y.view(-1).long()
                loss = criterion(out, true_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


            average_loss = total_loss / len(data_loader)
            print(f"Dataset {m}, Loss: {average_loss:.4f}")
            epoch_loss_total += average_loss


        avg_epoch_loss = epoch_loss_total / len(dataset_numbers)
        print(f"Epoch_{weight} {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
        all_epoch_losses.append(avg_epoch_loss)


        model.eval()
        all_total_true_positives = 0
        all_total_actual_positives = 0
        all_total_predicted_positives = 0

        with torch.no_grad():
            for n in dataset_numbers1:
                pkl_filename = f'temp1/ERR8562466/graphdata/dataset_{n}.pkl'

                with open(pkl_filename, 'rb') as f:
                    loaded_graphs = pickle.load(f)

                val_loader = DataLoader(loaded_graphs, batch_size=1, shuffle=True)

                total_true_positives = 0
                total_actual_positives = 0
                total_predicted_positives = 0

                for batch in val_loader:
                    outputs = model(batch)
                    predicted = torch.argmax(outputs, dim=1)
                    true_y = batch.y.view(-1).long()

                    batch_conf_matrix = confusion_matrix(true_y.cpu().numpy(), predicted.view(-1).cpu().numpy())
                    if batch_conf_matrix.shape[0] > 1 and batch_conf_matrix.shape[1] > 1:
                        tp = batch_conf_matrix[1, 1]
                        fn = batch_conf_matrix[1, 0]
                        fp = batch_conf_matrix[0, 1]

                        total_true_positives += tp
                        total_actual_positives += (tp + fn)
                        total_predicted_positives += (tp + fp)

                all_total_true_positives += total_true_positives
                all_total_actual_positives += total_actual_positives
                all_total_predicted_positives += total_predicted_positives


            overall_precision = all_total_true_positives / all_total_predicted_positives if all_total_predicted_positives > 0 else 0
            print(f'Overall Validation Precision (label 1)_{weight}: {overall_precision * 100:.2f}%')


            if overall_precision > best_precision:
                best_precision = overall_precision
                counter = 0
                torch.save(model.state_dict(), f'temp1/ERR8562466/model/best_model_{weight}.pth')
                print(f"New best model saved with precision_{weight}: {best_precision * 100:.2f}%")
            else:
                counter += 1
                print(f"Early Stopping Counter: {counter}/{patience}")


            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best precision: {best_precision * 100:.2f}%")
                break


    save_loss_path = f'temp1/ERR8562466/model/epoch_losses_{weight}.pkl'
    with open(save_loss_path, 'wb') as f:
        pickle.dump(all_epoch_losses, f)
