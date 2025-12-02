import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.kernel_size = 3
        self.pad = (self.kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(1, 16, self.kernel_size, padding='valid')
        self.conv2 = nn.Conv1d(16, 32, self.kernel_size, padding='valid')
        self.conv3 = nn.Conv1d(32, 64, self.kernel_size, padding='valid')

        self.pool1 = nn.AdaptiveMaxPool1d(300)
        self.pool2 = nn.AdaptiveMaxPool1d(150)
        self.pool3 = nn.AdaptiveMaxPool1d(64)

        self.fc1 = nn.Linear(64 * 64, 256)
        self.fc2 = nn.Linear(256, 500)
        self.relu = nn.ReLU()
   

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode='circular')                       
                                                           

        x = self.relu(self.conv1(x))                                  
        x = F.pad(x, (self.pad, self.pad), mode='circular')
                                                           

        x = self.pool1(x)                        


        x = F.pad(x, (self.pad, self.pad), mode='circular')                        
                                                           

        x = self.relu(self.conv2(x))                                  
        x = F.pad(x, (self.pad, self.pad), mode='circular')                        
                                                           

        x = self.pool2(x)                        


        x = F.pad(x, (self.pad, self.pad), mode='circular')                         
                                                           
        x = self.relu(self.conv3(x))                        

        x = F.pad(x, (self.pad, self.pad), mode='circular')                         
                                                           
        x = self.pool3(x)   

        x = x.view(x.size(0), -1)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
    

              
def Pre_data(csv_path, num):
    data = pd.read_csv(csv_path)
    x = data['x'].values
    u_data = data.drop('x', axis=1).values.T

                           
    u_min, u_max = u_data.min(), u_data.max()
    u_normalized = 2 * (u_data - u_min) / (u_max - u_min) - 1

                      
    X_list = []
    y_list = []
    for i in range(num):
        start = i * 201
        end = (i + 1) * 201
        u_exp = u_normalized[start:end]                    
        
        X_exp = u_exp[:-1]                    
        X_exp = X_exp[:, np.newaxis, :]                       

        y_exp = u_exp[1:] - u_exp[:-1]                   
        X_list.append(X_exp)
        y_list.append(y_exp)

    X_total = np.concatenate(X_list, axis=0)                                
    y_total = np.concatenate(y_list, axis=0)                          

    return X_total, y_total, x, u_min, u_max, u_normalized

                   
def create_dataloaders(X, y, batch_size):
    train_size = int(0.7 * len(y))
    val_size = int(0.15 * len(y))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_test, y_test

           
def train_model(model, train_loader, val_loader, device, num_epochs, save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainplot_loss = []
    valplot_loss = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)          
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 

        train_loss /= len(train_loader)
        trainplot_loss.append(train_loss)

        model.eval()
        val_loss = 0       
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            val_loss += criterion(outputs, y_batch).item()

        val_loss /= len(val_loader)
        valplot_loss.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.5f}, Val Loss: {val_loss/len(val_loader):.5f}")

    torch.save(model.state_dict(), save_path)
    return trainplot_loss, valplot_loss


            
def predict_at_time(model, u_initial, t_target, t_start, dt_frame, device, u_min, u_max, X_compare):
    model.eval()
    mse_errors = []
    predictions = []

    history = u_initial[np.newaxis, np.newaxis, :]              
    u_current = u_initial.copy()         
    t_current = t_start
    steps_needed = int((t_target - t_start) / dt_frame)
    times = [t_start + i * dt_frame for i in range(steps_needed)]

    if steps_needed <= 0:
        return None, None, None, None, None
    
    for step in range(steps_needed):
        X_tensor = torch.tensor(history, dtype=torch.float32).to(device)               
        delta_u = model(X_tensor).detach().cpu().numpy().squeeze()         
        u_current += delta_u
        t_current += dt_frame

        u_pred_denorm = (u_current + 1) * (u_max - u_min) / 2 + u_min
        predictions.append(u_pred_denorm.copy())

        u_true = (X_compare[step] + 1) * (u_max - u_min) / 2 + u_min
        mse = np.mean((u_pred_denorm - u_true) ** 2)
        mse_errors.append(mse)

        history = u_current[np.newaxis, np.newaxis, :]               

    return u_pred_denorm, t_current, predictions, mse_errors, times

def plot_comparison_value_at_time(t_compare, t_start, dt_frame, predictions, X, x, u_min, u_max, batch_size, num, idx):
    step = int((t_compare - t_start) / dt_frame) - 1

    u_true_denorm = (X[step] + 1) * (u_max - u_min) / 2 + u_min

    plt.figure(figsize=(10, 6))    
    plt.plot(x, predictions[step], label=f'Predicted u(x, t={t_compare:.2f})', linewidth=2)
    plt.plot(x, u_true_denorm, label=f'True u(x, t={t_compare:.2f})', linestyle='--', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Comparison at t={t_compare:.2f}, batch_size={batch_size}, exp={idx}, {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
    save_path = os.path.join(save_dir, f"wave_comparison_batch_{batch_size}_exp{idx}_t{t_compare:.2f}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_losses(train_losses_all, val_losses_all, num):
    plt.figure(figsize=(12, 6))
    batch_sizes = list(train_losses_all.keys())
    color_values = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    color_map = {bs: color_values[i] for i, bs in enumerate(batch_sizes)}

    for batch_size in batch_sizes:
        color = color_map[batch_size]
        plt.plot(range(1, len(train_losses_all[batch_size]) + 1), 
                 train_losses_all[batch_size], 
                 label=f'Train Loss (batch_size={batch_size})', linewidth=2, color=color)
        plt.plot(range(1, len(val_losses_all[batch_size]) + 1), 
                 val_losses_all[batch_size], 
                 label=f'Val Loss (batch_size={batch_size})', linestyle='--', linewidth=2, color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Training and Validation Loss for {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
    save_path = os.path.join(save_dir, f"losses_num{num}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
                

def plot_mse(mse_errors_all, times, num, compare_index):
    plt.figure(figsize=(10, 6))
    for batch_size in mse_errors_all:
        plt.plot(times[:len(mse_errors_all[batch_size])], 
                 mse_errors_all[batch_size], 
                 label=f'batch_size={batch_size}', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('MSE')
    plt.title(f'MSE at Time for exp={compare_index}, {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
    save_path = os.path.join(save_dir, f"mse_exp{compare_index}_num{num}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
                


def train_main():
    device = torch.device('cuda')
    print(f"Using device: {device}")

    csv_paths = [
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_45.csv"
    ]
    num_list = [45]
    batch_sizes = [32, 64, 128]

    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final"

    for csv_path, num in zip(csv_paths, num_list):
        print(f"\ndataset: {csv_path} ({num} rogue waves)")

        X_total, y_total, x, u_min, u_max, u_normalized = Pre_data(csv_path, num)

        train_losses_all = {batch_size: [] for batch_size in batch_sizes}
        val_losses_all = {batch_size: [] for batch_size in batch_sizes}

        min_train_loss = float('inf')
        best_batch_size = None

        for batch_size in batch_sizes:
            print(f"\n batch_size={batch_size}")

            train_loader, val_loader, X_test, y_test = create_dataloaders(X_total, y_total, batch_size)

            model = CNN().to(device)

            save_path = os.path.join(save_dir, f"model_batch_{batch_size}_exp{num}.pth")

            train_losses, val_losses = train_model(model, train_loader, val_loader, device, num_epochs=50, save_path=save_path)
            train_losses_all[batch_size] = train_losses
            val_losses_all[batch_size] = val_losses

            final_train_loss = train_losses[-1]
            
            if final_train_loss < min_train_loss:
                min_train_loss = final_train_loss
                best_batch_size = batch_size

        plot_losses(train_losses_all, val_losses_all, num)
        if best_batch_size is not None:
            print(f"Best batch size for {num} rogue waves: {best_batch_size}")


def plot_dataset_comparison(predictions_dict, true_values, x, batch_size, num_list, idx, t_compare):
    plt.figure(figsize=(12, 8))
    
    plt.plot(x, true_values, label='True', color='black', linewidth=3)
    
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', ':', '-.']
    
    for i, (num, pred) in enumerate(predictions_dict.items()):
        plt.plot(x, pred, 
                label=f'Predicted (Dataset {num})', 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Comparison of Predictions from Different Datasets at t={t_compare:.2f}\n'
              f'batch_size={batch_size}, exp={idx}')
    plt.legend()
    plt.grid(True)
    
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
    save_path = os.path.join(save_dir, f"dataset_comparison_batch_{batch_size}_exp{idx}_t{t_compare:.2f}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_individual_dataset_prediction(predictions, true_values, x, batch_size, num, idx, t_compare):
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, predictions, label=f'Predicted (Dataset {num})', color='blue', linewidth=2)
    plt.plot(x, true_values, label='True', color='black', linestyle='--', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'CNN prediction results, batch size = {batch_size}')
    plt.legend()
    plt.grid(True)
    
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
    save_path = os.path.join(save_dir, f"individual_dataset_{num}_batch_{batch_size}_exp{idx}_t{t_compare:.2f}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_batch_size_comparison(predictions_dict, true_values, x, num, idx, t_compare):
    plt.figure(figsize=(12, 8))

    plt.plot(x, true_values, label='True', color='black', linewidth=3)

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    linestyles = ['-', '--', ':', '-.', '-']
    
    for i, (batch_size, pred) in enumerate(predictions_dict.items()):
        plt.plot(x, pred, 
                label=f'Predicted (batch_size={batch_size})', 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Batch Size Comparison for Dataset {num} at t={t_compare:.2f}\nexp={idx}')
    plt.legend()
    plt.grid(True)
    
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
    save_path = os.path.join(save_dir, f"batch_size_comparison_dataset_{num}_exp{idx}_t{t_compare:.2f}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_comprehensive_comparison(all_predictions, true_values, x, num_list, batch_sizes, idx, t_compare, best_batch_sizes_dict=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Comprehensive Comparison at t={t_compare:.2f}, exp={idx}', fontsize=16)

    dataset_colors = {'15': 'blue', '30': 'red', '45': 'green'}
    batch_colors = {'32': 'blue', '64': 'red', '128': 'green'}

                                                              
    ax1 = axes[0]
    ax1.plot(x, true_values, label='True', color='black', linewidth=3)
    colors_repr = ['blue', 'red', 'green']
    
    if best_batch_sizes_dict:
        for i, num in enumerate(num_list):
            best_batch = best_batch_sizes_dict[num]
            if (num, best_batch) in all_predictions:
                pred = all_predictions[(num, best_batch)]
                ax1.plot(x, pred, 
                        label=f'Dataset {num}, batch {best_batch} (optimal)', 
                        color=colors_repr[i],
                        linewidth=2)
    else:
                                                                         
        representative_combinations = [(15, 32), (30, 64), (45, 128)]
        for i, (num, batch_size) in enumerate(representative_combinations):
            if (num, batch_size) in all_predictions:
                pred = all_predictions[(num, batch_size)]
                ax1.plot(x, pred, 
                        label=f'Dataset {num}, batch {batch_size}', 
                        color=colors_repr[i],
                        linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('Dataset Size Comparison (Optimal Batch Sizes)')
    ax1.grid(True)
    ax1.legend()
    
                                                           
    ax2 = axes[1]
    ax2.plot(x, true_values, label='True', color='black', linewidth=3)
    largest_dataset = max(num_list)                           
    for batch_size in batch_sizes:
        if (largest_dataset, batch_size) in all_predictions:
            pred = all_predictions[(largest_dataset, batch_size)]
            ax2.plot(x, pred, 
                    label=f'batch_size={batch_size}', 
                    color=batch_colors[str(batch_size)],
                    linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u')
    ax2.set_title(f'Batch Size Comparison (Dataset {largest_dataset})')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
    save_path = os.path.join(save_dir, f"comprehensive_comparison_exp{idx}_t{t_compare:.2f}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def predict_main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    csv_paths = [
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_1.csv",
    ]

    num_list = [15, 30, 45] 
    exp_indices = [0]  
    t_start = 0.0
    t_target = 3.0
    t_compare = 2.8
    dt_frame = 0.1
    batch_sizes = [32, 64, 128]

    for csv_path, idx in zip(csv_paths, exp_indices):
        print(f"\ndataset: {csv_path} (exp={idx+1})")

        all_predictions = {}
        all_mse_errors = {}
        all_times = {}

        best_batch_sizes_dict = {}

        _, _, x, u_min, u_max, u_normalized = Pre_data(csv_path, 1)
        index_start = 70 + idx * 201
        index_target = 101 + idx * 201
        u_initial = u_normalized[index_start] 
        X_compare = u_normalized[index_start:index_target]
        step = int((t_compare - t_start) / dt_frame) - 1
        u_true_denorm = (X_compare[step] + 1) * (u_max - u_min) / 2 + u_min
        
        for num in num_list:
            print(f"\nPredicting with models trained on {num} rogue waves dataset")

            batch_predictions = {}
            batch_final_mse = {}
            
            for batch_size in batch_sizes:
                print(f"  batch_size={batch_size}")
            
            model = CNN().to(device)
            save_path = f"C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final/model_batch_{batch_size}_exp{num}.pth"
            model.load_state_dict(torch.load(save_path))

            u_pred_denorm, t_pred, predictions, mse_errors, times = predict_at_time(
                model, u_initial, t_target, t_start, dt_frame, device, u_min, u_max, X_compare
            )
            
            all_predictions[(num, batch_size)] = predictions[step]
            all_mse_errors[(num, batch_size)] = mse_errors
            all_times[(num, batch_size)] = times
            batch_predictions[batch_size] = predictions[step]
                
            batch_final_mse[batch_size] = mse_errors[-1] if mse_errors else float('inf')
            
            print(f"Prediction completed for dataset {num}, batch_size={batch_size}, t={t_pred:.2f}, final_MSE={batch_final_mse[batch_size]:.6f}")

            plot_individual_dataset_prediction(predictions[step], u_true_denorm, x, batch_size, num, idx + 1, t_compare)
            
            best_batch_size = min(batch_final_mse.keys(), key=lambda k: batch_final_mse[k])
            best_batch_sizes_dict[num] = best_batch_size
            print(f"  Best batch_size for dataset {num}: {best_batch_size} (MSE: {batch_final_mse[best_batch_size]:.6f})")
            
            plot_batch_size_comparison(batch_predictions, u_true_denorm, x, num, idx + 1, t_compare)

        dataset_predictions = {}
        for num in num_list:
            best_batch = best_batch_sizes_dict[num]
            if (num, best_batch) in all_predictions:
                dataset_predictions[num] = all_predictions[(num, best_batch)]
        plot_dataset_comparison(dataset_predictions, u_true_denorm, x, "best", num_list, idx + 1, t_compare)
        
        plot_comprehensive_comparison(all_predictions, u_true_denorm, x, num_list, batch_sizes, idx + 1, t_compare, best_batch_sizes_dict)
        
        for num in num_list:
            plt.figure(figsize=(12, 6))
            colors = ['blue', 'red', 'green', 'purple']
            for i, batch_size in enumerate(batch_sizes):
                if (num, batch_size) in all_mse_errors:
                    line_style = '-' if batch_size == best_batch_sizes_dict[num] else '--'
                    line_width = 3 if batch_size == best_batch_sizes_dict[num] else 2
                    label = f'batch_size={batch_size}' + (' (best)' if batch_size == best_batch_sizes_dict[num] else '')
                    plt.plot(all_times[(num, batch_size)], all_mse_errors[(num, batch_size)], 
                            label=label, 
                            color=colors[i % len(colors)],
                            linewidth=line_width,
                            linestyle=line_style)
            plt.xlabel('Time t')
            plt.ylabel('MSE')
            plt.title(f'MSE Comparison for Dataset {num} Across Batch Sizes\nexp={idx + 1}')
            plt.legend()
            plt.grid(True)
            save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
            save_path = os.path.join(save_dir, f"mse_batch_comparison_dataset_{num}_exp{idx + 1}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()

        for batch_size in batch_sizes:
            plt.figure(figsize=(12, 6))
        colors = ['blue', 'red', 'green', 'purple']
        for i, num in enumerate(num_list):
                if (num, batch_size) in all_mse_errors:
                    line_style = '-' if batch_size == best_batch_sizes_dict[num] else '--'
                    line_width = 3 if batch_size == best_batch_sizes_dict[num] else 2
                    label = f'Dataset {num}' + (' (best for this dataset)' if batch_size == best_batch_sizes_dict[num] else '')
                    plt.plot(all_times[(num, batch_size)], all_mse_errors[(num, batch_size)], 
                            label=label, 
                    color=colors[i % len(colors)],
                            linewidth=line_width,
                            linestyle=line_style)
        plt.xlabel('Time t')
        plt.ylabel('MSE')
        plt.title(f'MSE Comparison for batch_size={batch_size} Across Datasets\nexp={idx + 1}')
        plt.legend()
        plt.grid(True)
        save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_final_results"
        save_path = os.path.join(save_dir, f"mse_dataset_comparison_batch_{batch_size}_exp{idx + 1}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
            
        print(f"\n=== Best Batch Sizes Summary for exp={idx + 1} ===")
        for num in num_list:
            print(f"Dataset {num}: batch_size={best_batch_sizes_dict[num]}")
        print("=" * 50)

if __name__ == '__main__':
    train_main()
    predict_main()

