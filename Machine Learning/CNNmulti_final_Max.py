import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import math

                              
                                       
                              
def compute_rmse_over_domain(truth, estimate):
    diff = estimate - truth
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_peak_amplitude_error(truth, estimate, region=None):
    if region is not None and region.shape == truth.shape:
        truth_region = truth[region]
        est_region = estimate[region]
    else:
        truth_region = truth
        est_region = estimate
    if truth_region.size == 0 or est_region.size == 0:
        return float('nan')
    return float(np.abs(np.max(est_region) - np.max(truth_region)))


def compute_mean_abs_error_region(truth, estimate, region):
    if region is None:
        return float('nan')
    if region.shape != truth.shape:
        return float('nan')
    if not np.any(region):
        return float('nan')
    return float(np.mean(np.abs(estimate[region] - truth[region])))

     
class CNN(nn.Module):
    def __init__(self, k, m, m_max, kernel_size=3):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.pad = (self.kernel_size - 1) // 2
        self.m = m
        self.m_max = m_max
        self.conv1 = nn.Conv1d(k, 16, 3, padding='valid')
        self.conv2 = nn.Conv1d(16, 32, 3, padding='valid')
        self.conv3 = nn.Conv1d(32, 64, 3, padding='valid')
        self.pool1 = nn.AdaptiveMaxPool1d(300)
        self.pool2 = nn.AdaptiveMaxPool1d(150)
        self.pool3 = nn.AdaptiveMaxPool1d(64)
        self.fc1 = nn.Linear(64 * 64, 256)
        self.fc2 = nn.Linear(256, m_max * 500)
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
        x = x.view(x.size(0), self.m_max, 500)                            
        x = x[:, :self.m, :]                         

        return x

    def get_layer_outputs(self, x):
        outputs = {}
               
        x1 = F.pad(x, (self.pad, self.pad), mode='circular')
        x1 = self.relu(self.conv1(x1))
        outputs['conv1'] = x1
        x1p = self.pool1(x1)
        outputs['pool1'] = x1p
               
        x2 = F.pad(x1p, (self.pad, self.pad), mode='circular')
        x2 = self.relu(self.conv2(x2))
        outputs['conv2'] = x2
        x2p = self.pool2(x2)
        outputs['pool2'] = x2p
               
        x3 = F.pad(x2p, (self.pad, self.pad), mode='circular')
        x3 = self.relu(self.conv3(x3))
        outputs['conv3'] = x3
        x3p = self.pool3(x3)
        outputs['pool3'] = x3p
             
        x_flat = x3p.view(x3p.size(0), -1)
        x_fc1 = self.relu(self.fc1(x_flat))
        outputs['fc1'] = x_fc1
             
        x_fc2 = self.fc2(x_fc1)
        x_fc2 = x_fc2.view(x_fc2.size(0), self.m_max, 500)
        outputs['fc2'] = x_fc2
                      
        outputs['final'] = x_fc2[:, :self.m, :]
        return outputs

def calculate_layer_wise_cost(model, X_batch, y_batch, m, device):
    model.eval()
    with torch.no_grad():
        costs = {}
        layer_outputs = model.get_layer_outputs(X_batch)
        
        for layer_name, output in layer_outputs.items():
            if layer_name != 'final':
                costs[layer_name] = F.mse_loss(output, torch.zeros_like(output)).item()
        
        final_output = layer_outputs['final']
        for i in range(m):
            step_output = final_output[:, i, :]
            step_target = y_batch[:, i, :]
            costs[f'step_{i+1}'] = F.mse_loss(step_output, step_target).item()
        
        return costs

def plot_layer_wise_performance(costs_history, save_dir, dataset_name=""):
    layer_names = [k for k in costs_history[0].keys() if k in ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'fc1']]
    n_layers = len(layer_names)
    n_cols = 2
    n_rows = math.ceil(n_layers / n_cols)
    plt.figure(figsize=(12, 2.5 * n_rows))
    stages = range(1, len(costs_history) + 1)
    for i, layer in enumerate(layer_names):
        plt.subplot(n_rows, n_cols, i + 1)
        stage_costs = [stage[layer] for stage in costs_history]
        plt.plot(stages, stage_costs, marker='o')
        plt.title(f'{layer} Cost Evolution')
        plt.xlabel('Stage (m)')
        plt.ylabel('Cost')
        plt.grid(True)
        for j, cost in enumerate(stage_costs):
            plt.text(j+1, cost, f'{cost:.6f}', ha='center', va='bottom')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'layer_performance_{dataset_name}.png')
    plt.savefig(save_path)
    plt.close()





def plot_multi_dataset_costs(dataset_costs_history, num_list, save_dir):
    layer_names = [k for k in dataset_costs_history[0][0].keys() if k in ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'fc1']]
    stages = range(1, len(dataset_costs_history[0]) + 1)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    
    for layer in layer_names:
        plt.figure(figsize=(8, 5))
        for i, costs_history in enumerate(dataset_costs_history):
            costs = [stage[layer] for stage in costs_history]
            plt.plot(stages, costs, marker='o', label=f'Dataset {num_list[i]}', color=colors[i % len(colors)])
        plt.xlabel('Stage (m)')
        plt.ylabel('Cost')
        plt.title(f'{layer} Cost Evolution Across Datasets')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'multi_dataset_{layer}_cost.png')
        plt.savefig(save_path)
        plt.close()



def pre_data(csv_path, num, k, m):
    data = pd.read_csv(csv_path)
    x = data['x'].values
    u_data = data.drop('x', axis=1).values.T                      

                           
    u_min, u_max = u_data.min(), u_data.max()
    u_normalized = 2 * (u_data - u_min) / (u_max - u_min) - 1

    X_list = []
    y_list = []
    for i in range(num):
        start_col = i * 201
        end_col = (i + 1) * 201
        u_exp = u_normalized[start_col:end_col]               

        X_exp = []
        y_exp = []
        for j in range(len(u_exp) - k - m + 1):
            X_exp.append(u_exp[j:j+k])
            deltas = []
            for t in range(m):
                delta = u_exp[j+k+t] - u_exp[j+k+t-1]
                deltas.append(delta)
            y_exp.append(deltas)
        X_exp = np.array(X_exp)
        y_exp = np.array(y_exp)

        X_list.append(X_exp)
        y_list.append(y_exp)

    X_total = np.concatenate(X_list, axis=0)                                 
    y_total = np.concatenate(y_list, axis=0)                                 
    print(f"X_total shape: {X_total.shape}, y_total shape: {y_total.shape}")
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

def train_model(model, train_loader, val_loader, device, num_epochs, k, m, save_path, prev_model_path=None, stage_params_dir=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainplot_loss = []
    valplot_loss = []
    costs_history = []

    if prev_model_path and os.path.exists(prev_model_path):
        print(f"Loading previous model from {prev_model_path}")
        prev_state = torch.load(prev_model_path)
        current_state = model.state_dict()
        compatible_state = {
            k: v for k, v in prev_state.items()
            if k in current_state and v.shape == current_state[k].shape
        }
        model.load_state_dict(compatible_state, strict=False)
        print(f"Loaded {len(compatible_state)} / {len(current_state)} parameters")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(X_batch)
            y_target = y_batch[:, :m, :]
            
            loss = criterion(outputs, y_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        trainplot_loss.append(train_loss)

                                    
        if epoch == num_epochs - 1:                                     
            model.eval()
            with torch.no_grad():
                X_batch, y_batch = next(iter(val_loader))
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                step_costs = calculate_layer_wise_cost(model, X_batch, y_batch, m, device)
                costs_history.append(step_costs)

                     
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                y_target = y_batch[:, :m, :]
                val_loss += criterion(outputs, y_target).item()

        val_loss /= len(val_loader)
        valplot_loss.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, k={k}, m={m}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    return trainplot_loss, valplot_loss, costs_history

def predict_at_time(model, u_initial, t_target, t_start, dt_frame, device, u_min, u_max, X_compare, k, m):
    model.eval()
    mse_errors = []
    predictions = []
    true_values = []
    metrics_records = []

    history = u_initial[-k:]            
    u_current = u_initial[-1].copy()         
    t_current = t_start
    
    steps_needed = int((t_target - t_start) / dt_frame)
    times = [t_start + i * dt_frame for i in range(steps_needed)]

    if steps_needed <= 0:
        return None, None, None, None, None, None

    step = 0
    while step < steps_needed:
        X_tensor = torch.tensor(history[np.newaxis, :, :], dtype=torch.float32).to(device)
        delta_us = model(X_tensor).detach().cpu().numpy().squeeze()            
        if delta_us.ndim == 1:
            delta_us = delta_us[np.newaxis, :]
        
        for i in range(min(m, steps_needed - step)):
            delta_u = delta_us[i]
            u_current += delta_u
            t_current += dt_frame
            history = np.vstack([history[1:], u_current])
            
            u_pred_denorm = (u_current + 1) * (u_max - u_min) / 2 + u_min
            predictions.append(u_pred_denorm.copy())
            
            u_true_denorm = (X_compare[step + i] + 1) * (u_max - u_min) / 2 + u_min
            true_values.append(u_true_denorm.copy())
            
            mse = np.mean((u_pred_denorm - u_true_denorm) ** 2)
            mse_errors.append(mse)

                                            
            rmse_domain = compute_rmse_over_domain(u_true_denorm, u_pred_denorm)
            rogue_threshold = 1.1
            region = u_true_denorm > rogue_threshold
            if not np.any(region):
                region = None
            peak_amp_error = compute_peak_amplitude_error(u_true_denorm, u_pred_denorm, region)
            mean_abs_err_peak_region = compute_mean_abs_error_region(u_true_denorm, u_pred_denorm, region)
            metrics_records.append({
                'time_s': float(t_current),
                'rmse_domain': rmse_domain,
                'peak_amplitude_error': peak_amp_error,
                'mean_abs_error_peak_region': mean_abs_err_peak_region
            })
            
            print(f"Step {step + i}: pred t={t_current:.2f}, true t={t_start + (step + i + 1)*dt_frame:.2f}, "
                  f"pred mean={u_pred_denorm.mean():.4f}, true mean={u_true_denorm.mean():.4f}, "
                  f"pred min={u_pred_denorm.min():.4f}, pred max={u_pred_denorm.max():.4f}, "
                  f"true min={u_true_denorm.min():.4f}, true max={u_true_denorm.max():.4f}")
        
        step += m

    return u_pred_denorm, t_current, predictions, mse_errors, times, true_values, metrics_records

def plot_comparison_value_at_time(t_compare, t_start, dt_frame, predictions, true_values, x, batch_size, num, idx, k, m):
    step = int((t_compare - t_start) / dt_frame) - 1
    plt.figure(figsize=(10, 6))    
    plt.plot(x, predictions[step], label=f'Predicted u(x, t={t_compare:.2f})', linewidth=2)
    plt.plot(x, true_values[step], label=f'True u(x, t={t_compare:.2f})', linestyle='--', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Comparison at t={t_compare:.2f}, batch_size={batch_size}, m={m}, {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    save_path = os.path.join(save_dir, f"wave_comparison_batch_{batch_size}_exp{idx}_num{num}_t{t_compare:.2f}_k={k}_m={m}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_comparison_m_values(t_compare, t_start, dt_frame, predictions_dict, true_values_dict, x, batch_size, num, idx, k, m_values):
    step = int((t_compare - t_start) / dt_frame) - 1
    plt.figure(figsize=(10, 6))    
    plt.plot(x, true_values_dict[m_values[0]][step], label=f'True u(x, t={t_compare:.2f})', linestyle='--', color='black', linewidth=2)
    colors = ['blue', 'red', 'green', 'purple']                                 
    for m, color in zip(m_values, colors):
        predictions = predictions_dict[m]
        plt.plot(x, predictions[step], label=f'Predicted (m={m})', color=color, linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Comparison of Different m Values at t={t_compare:.2f}, batch_size={batch_size}, {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    save_path = os.path.join(save_dir, f"wave_comparison_m_values_batch_{batch_size}_exp{idx}_num{num}_t{t_compare:.2f}_k={k}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_losses(train_losses_all, val_losses_all, num, k, m):
    plt.figure(figsize=(12, 6))
    for batch_size in train_losses_all:
        plt.plot(range(1, len(train_losses_all[batch_size]) + 1), 
                 train_losses_all[batch_size], 
                 label=f'Train Loss (batch_size={batch_size})', linewidth=2)
        plt.plot(range(1, len(val_losses_all[batch_size]) + 1), 
                 val_losses_all[batch_size], 
                 label=f'Val Loss (batch_size={batch_size})', linestyle='--', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {num} Rogue Wave Experiments, m={m}')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    save_path = os.path.join(save_dir, f"losses_num{num}_k={k}_m={m}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_mse(mse_errors_all, times, num, compare_index, k, m):
    plt.figure(figsize=(10, 6))
    for batch_size in mse_errors_all:
        plt.plot(times[:len(mse_errors_all[batch_size])], 
                 mse_errors_all[batch_size], 
                 label=f'batch_size={batch_size}', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('MSE')
    plt.title(f'MSE at Time, m={m}, {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    save_path = os.path.join(save_dir, f"mse_exp{compare_index}_num{num}_k={k}_m={m}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_mse_m_values(mse_errors_dict, times, num, compare_index, batch_size, k, m_values):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'purple']                                 
    for m, color in zip(m_values, colors):
        mse_errors = mse_errors_dict[m]
        plt.plot(times[:len(mse_errors)], mse_errors, label=f'm={m}', color=color, linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('MSE')
    plt.title(f'MSE Comparison for Different m Values, batch_size={batch_size}, {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    save_path = os.path.join(save_dir, f"mse_comparison_m_values_batch_{batch_size}_exp{compare_index}_num{num}_k={k}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_best_losses(best_train_losses, best_batch_sizes, num_list, m_values):
    plt.figure(figsize=(12, 6))
    for i, (train_losses, batch_size, num, m) in enumerate(zip(best_train_losses, best_batch_sizes, num_list, m_values)):
        plt.plot(range(1, len(train_losses) + 1), 
                 train_losses, 
                 label=f'noise_{num}, batch_size={batch_size}, m={m}', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Best Training Loss Comparison Across Datasets')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    save_path = os.path.join(save_dir, f"best_losses.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_curriculum_mse_by_batch_size(mse_dict, num_exp, m_values):
    for batch_size in set(bs for bs, m in mse_dict.keys()):
        plt.figure(figsize=(12, 6))
        m_values = sorted([m for (bs, m) in mse_dict.keys() if bs == batch_size])
        for idx, m in enumerate(m_values):
            train_mse = mse_dict[(batch_size, m)]
            x_range = range(idx * 50 + 1, (idx + 1) * 50 + 1)
            plt.plot(x_range, train_mse, label=f'm={m}', linewidth=2)
            if idx < len(m_values) - 1:
                plt.axvline(x=(idx + 1) * 50, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Extended Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'Curriculum Learning MSE for {num_exp} Rogue Wave Experiments, batch_size={batch_size}')
        plt.legend()
        plt.grid(True)
        save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
        save_path = os.path.join(save_dir, f"curriculum_mse_batch_{batch_size}_num{num_exp}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

def plot_train_mse_m_values(mse_dict, num, batch_size, m_values):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'purple']                                 
    for m, color in zip(m_values, colors):
        mse_errors = mse_dict[m]
        plt.plot(range(1, len(mse_errors) + 1), mse_errors, label=f'm={m}', color=color, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Train MSE')
    plt.title(f'Train MSE Comparison for Different m Values, batch_size={batch_size}, {num} Rogue Wave Experiments')
    plt.legend()
    plt.grid(True)
    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    save_path = os.path.join(save_dir, f"train_mse_comparison_m_values_batch_{batch_size}_num{num}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_concat_layerwise_costs(dataset_costs_history, save_dir):
    all_costs = []
    for costs_history in dataset_costs_history:
        all_costs.extend(costs_history)
    n_stage = len(all_costs)
    layer_names = [k for k in all_costs[0].keys() if k in ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'fc1']]
    n_layers = len(layer_names)
    n_cols = 2
    n_rows = math.ceil(n_layers / n_cols)
    plt.figure(figsize=(12, 2.5 * n_rows))
    stages = range(1, n_stage + 1)
    seg_len = len(all_costs) // len(dataset_costs_history)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    dataset_labels = [f'Dataset {15 + 15*j}' for j in range(len(dataset_costs_history))]
    for i, layer in enumerate(layer_names):
        plt.subplot(n_rows, n_cols, i + 1)
        for j, costs_history in enumerate(dataset_costs_history):
            start = j * seg_len
            end = (j + 1) * seg_len
            stage_costs = [stage[layer] for stage in all_costs[start:end]]
            plt.plot(range(start+1, end+1), stage_costs, marker='o', color=colors[j % len(colors)], label=dataset_labels[j])
        plt.title(f'{layer} Cost Evolution')
        plt.xlabel('Stage')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'concat_layer_costs.png')
    plt.savefig(save_path)
    plt.close()



def compare_stage_costs_at_step4(model_paths, test_data, k, m_values, device='cuda'):

    X_test, y_test = test_data
    results = {}

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    for m in m_values:
        print(f"Testing stage m={m} parameters for step-4 prediction")
        model = CNN(k=k, m=m, m_max=max(m_values)).to(device)
        model.load_state_dict(torch.load(model_paths[m]))
        model.eval()
        
        mse_errors = []
        step4_predictions = []
        step4_true_values = []

        with torch.no_grad():
            num_samples = len(X_test)
            
            for i in range(num_samples):

                X_sample = X_test[i:i+1]               
                base_state = X_sample[0, -1, :].copy()
                
                X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)

                predictions = model(X_tensor)               
                predictions = predictions.detach().cpu().numpy()               

                if m >= 4:
                    pred_step4 = base_state.copy()
                    for step_idx in range(4):
                        pred_step4 += predictions[0, step_idx, :]
                else:
                    pred_step4 = base_state.copy()
                    history = X_sample[0].copy()            
                    
                    steps_predicted = 0
                    while steps_predicted < 4:
                        history_tensor = torch.tensor(history[np.newaxis, :, :], dtype=torch.float32).to(device)
                        step_predictions = model(history_tensor)               
                        step_predictions = step_predictions.detach().cpu().numpy()               
                        
                        steps_to_use = min(m, 4 - steps_predicted)
                        
                        for step_idx in range(steps_to_use):
                            delta = step_predictions[0, step_idx, :]
                            pred_step4 += delta
                            
                                                            
                            history = np.vstack([history[1:], pred_step4])
                            steps_predicted += 1
                    
                if i < len(y_test):
                    true_step4 = base_state.copy()
                    available_steps = min(4, y_test.shape[1])
                    for step_idx in range(available_steps):
                        true_step4 += y_test[i, step_idx, :]
                    if 4 > y_test.shape[1] and y_test.shape[1] > 0:
                        last_delta = y_test[i, -1, :]
                        for _ in range(y_test.shape[1], 4):
                            true_step4 += last_delta

                    mse = np.mean((pred_step4 - true_step4) ** 2)
                    mse_errors.append(mse)
                    step4_predictions.append(pred_step4.copy())
                    step4_true_values.append(true_step4.copy())
        

        avg_mse = np.mean(mse_errors) if mse_errors else float('inf')
        results[m] = {
            'avg_mse': avg_mse,
            'mse_errors': mse_errors,
            'predictions': step4_predictions,
            'true_values': step4_true_values
        }
        print(f"  Stage m={m} average MSE for step-4 prediction: {avg_mse:.6f}")
    
    return results

def compare_stage_costs_at_multiple_steps(model_paths, test_data, k, m_values, steps_to_predict=[4, 8, 12, 16, 20, 24, 28], device='cuda'):

    X_test, y_test = test_data
    results = {}
    
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    for m in m_values:
        print(f"Testing stage m={m} parameters for multi-step prediction")
        model = CNN(k=k, m=m, m_max=max(m_values)).to(device)
        model.load_state_dict(torch.load(model_paths[m]))
        model.eval()
        
        stage_results = {}
        
        for target_step in steps_to_predict:
            print(f"  Testing step-{target_step} prediction")
            mse_errors = []
            step_predictions = []
            step_true_values = []
            
            with torch.no_grad():
                num_samples = len(X_test)
                
                for i in range(num_samples):
                    X_sample = X_test[i:i+1]               
                    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)

                    current_state = X_sample[0, -1, :].copy()  
                    history = X_sample[0].copy()            

                    steps_predicted = 0
                    while steps_predicted < target_step:
                        history_tensor = torch.tensor(history[np.newaxis, :, :], dtype=torch.float32).to(device)
                        predictions = model(history_tensor)               
                        predictions = predictions.detach().cpu().numpy()               
                        
                        steps_to_use = min(m, target_step - steps_predicted)
                        
                        for step_idx in range(steps_to_use):
                            delta = predictions[0, step_idx, :]
                            current_state += delta

                            history = np.vstack([history[1:], current_state])
                            steps_predicted += 1
                    
                    pred_step = current_state

                    if i < len(y_test):
                        true_step = X_sample[0, -1, :].copy()
                        for step_idx in range(min(target_step, y_test.shape[1])):
                            true_step += y_test[i, step_idx, :]

                        if target_step > y_test.shape[1]:
                            last_delta = y_test[i, -1, :]
                            for step_idx in range(y_test.shape[1], target_step):
                                true_step += last_delta
                        
                        mse = np.mean((pred_step - true_step) ** 2)
                        mse_errors.append(mse)
                        step_predictions.append(pred_step.copy())
                        step_true_values.append(true_step.copy())
            
            avg_mse = np.mean(mse_errors) if mse_errors else float('inf')
            stage_results[target_step] = {
                'avg_mse': avg_mse,
                'mse_errors': mse_errors,
                'predictions': step_predictions,
                'true_values': step_true_values
            }
            print(f"Step-{target_step} average MSE: {avg_mse:.6f}")
        
        results[m] = stage_results
    
    return results

def plot_step4_stage_comparison(results, save_dir, dataset_name=""):
    m_values = list(results.keys())
    mse_values = [results[m]['avg_mse'] for m in m_values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f'Stage m={m}' for m in m_values], mse_values, 
                   color=['blue', 'orange', 'green', 'red'], alpha=0.7)

    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.2e}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Training Stage')
    plt.ylabel('Average MSE for Step-4 Prediction')
    plt.title(f'Step-4 Prediction Performance Comparison Across Training Stages {dataset_name}')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'step4_stage_comparison{dataset_name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(m_values, mse_values, marker='o', linewidth=3, markersize=10, color='red')
    
    for m, mse in zip(m_values, mse_values):
        plt.text(m, mse, f'{mse:.2e}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Training Stage (m)')
    plt.ylabel('Average MSE for Step-4 Prediction')
    plt.title(f'Step-4 Prediction Performance Evolution {dataset_name}')
    plt.grid(True)

    plt.xticks(m_values, [f'm={m}' for m in m_values])
       
    save_path = os.path.join(save_dir, f'step4_stage_evolution{dataset_name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    data = [[f'm={m}', f'{results[m]["avg_mse"]:.2e}'] for m in m_values]
    
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=data, 
                     colLabels=['Training Stage', 'Step-4 MSE'], 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title(f'Step-4 Prediction MSE Values {dataset_name}')
    save_path = os.path.join(save_dir, f'step4_mse_table{dataset_name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_multiple_steps_stage_comparison(results, save_dir, dataset_name=""):
    m_values = list(results.keys())
    steps_to_predict = list(results[m_values[0]].keys())

    for step in steps_to_predict:
        mse_values = [results[m][step]['avg_mse'] for m in m_values]

        plt.figure(figsize=(10, 6))
        plt.plot(m_values, mse_values, marker='o', linewidth=3, markersize=10, color='red')

        for m, mse in zip(m_values, mse_values):
            plt.text(m, mse, f'{mse:.2e}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Training Stage (m)')
        plt.ylabel(f'Average MSE for Step-{step} Prediction')
        plt.title(f'Step-{step} Prediction Performance Evolution Across Training Stages {dataset_name}')
        plt.grid(True)
        plt.xticks(m_values, [f'm={m}' for m in m_values])
               
        save_path = os.path.join(save_dir, f'step{step}_stage_evolution{dataset_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        bars = plt.bar([f'Stage m={m}' for m in m_values], mse_values, 
                       color=['blue', 'orange', 'green', 'red'], alpha=0.7)

        for bar, mse in zip(bars, mse_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mse:.2e}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Training Stage')
        plt.ylabel(f'Average MSE for Step-{step} Prediction')
        plt.title(f'Step-{step} Prediction Performance Comparison Across Training Stages {dataset_name}')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, f'step{step}_stage_comparison{dataset_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 4))
        data = [[f'm={m}', f'{results[m][step]["avg_mse"]:.2e}'] for m in m_values]
        
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=data, 
                         colLabels=['Training Stage', f'Step-{step} MSE'], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title(f'Step-{step} Prediction MSE Values {dataset_name}')
        save_path = os.path.join(save_dir, f'step{step}_mse_table{dataset_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', '*', 'x']
    
    for i, step in enumerate(steps_to_predict):
        mse_values = [results[m][step]['avg_mse'] for m in m_values]
        plt.plot(m_values, mse_values, marker=markers[i], linewidth=2, 
                markersize=8, label=f'Step {step}', color=colors[i])
        
        for m, mse in zip(m_values, mse_values):
            plt.text(m, mse, f'{mse:.2e}', ha='center', va='bottom', 
                    fontsize=8, color=colors[i])
    
    plt.xlabel('Training Stage (m)')
    plt.ylabel('Average MSE')
    plt.title(f'Multi-Step Prediction Performance Across Training Stages {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(m_values, [f'm={m}' for m in m_values])
    
    save_path = os.path.join(save_dir, f'multi_steps_stage_comparison{dataset_name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def train_main():
    device = torch.device('cuda')
    print(f"Using device: {device}")

    csv_paths = [
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_15.csv",     
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_30.csv",
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_45.csv"
    ]
    num_list = [15, 30, 45]
    batch_sizes_list = [
        [32, 64, 128], 
        [32, 64, 128],
        [32, 64, 128]
    ]

    k = 64
    m_values = [1, 2, 3, 4]                              
    m_max = max(m_values)                           
    num_epochs = 50

    best_train_losses = []
    best_batch_sizes = []
    best_m_values = []
    mse_dict = {}

    save_dir = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
    
    dataset_costs_history = []

    for csv_path, num, batch_sizes in zip(csv_paths, num_list, batch_sizes_list):
        print(f"\ndataset: {csv_path} ({num} rogue waves)")

        train_losses_all_m = {batch_size: {m: [] for m in m_values} for batch_size in batch_sizes}
        dataset_costs = [] 

        prev_model_path = None
        for m_idx, m in enumerate(m_values):
            print(f"\nCurriculum Learning: k={k}, m={m}")

            X_total, y_total, x, u_min, u_max, u_normalized = pre_data(csv_path, num, k, m)

            print(f"X_total shape: {X_total.shape}")                             
            print(f"y_total shape: {y_total.shape}")                             

            train_losses_all = {batch_size: [] for batch_size in batch_sizes}
            val_losses_all = {batch_size: [] for batch_size in batch_sizes}

            min_train_loss = float('inf')
            best_batch_size = None
            best_train_loss_curve = None
            best_costs_history = None

            for batch_size in batch_sizes:
                print(f"\n batch_size={batch_size}")

                train_loader, val_loader, X_test, y_test = create_dataloaders(X_total, y_total, batch_size)

                prev_model_path = None
                if m_idx > 0:
                    prev_m = m_values[m_idx-1]
                    prev_model_path = f"C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_v2/model_batch_{batch_size}_exp{num}_k{k}_m{prev_m}.pth"
                model = CNN(k=k, m=m, m_max=m_max).to(device)

                save_path = f"C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_v2/model_batch_{batch_size}_exp{num}_k{k}_m{m}.pth"

                train_losses, val_losses, costs_history = train_model(
                    model, train_loader, val_loader, device, 
                    num_epochs=50, k=k, m=m,
                    save_path=save_path, 
                    prev_model_path=prev_model_path,
                    stage_params_dir=save_dir
                )
                train_losses_all[batch_size] = train_losses
                val_losses_all[batch_size] = val_losses
                mse_dict[(batch_size, m)] = train_losses

                train_losses_all_m[batch_size][m] = train_losses

                final_train_loss = train_losses[-1]
                
                if final_train_loss < min_train_loss:
                    min_train_loss = final_train_loss
                    best_batch_size = batch_size
                    best_train_loss_curve = train_losses
                    best_costs_history = costs_history

            plot_losses(train_losses_all, val_losses_all, num, k, m_values)

            best_train_losses.append(best_train_loss_curve)
            best_batch_sizes.append(best_batch_size)
            best_m_values.append(m)

            prev_model_path = f"C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_v2/model_batch_{best_batch_size}_exp{num}_k{k}_m{m}.pth"

            
            if best_costs_history:
                dataset_costs.extend(best_costs_history)
        
        if dataset_costs:  
            dataset_costs_history.append(dataset_costs)
        
        for batch_size in batch_sizes:
            mse_errors_dict = {m: train_losses_all_m[batch_size][m] for m in m_values}
            plot_train_mse_m_values(mse_errors_dict, num, batch_size, m_values)
        
        plot_curriculum_mse_by_batch_size(mse_dict, num, m_values)

        model_paths = {}
        for m in m_values:
            model_paths[m] = f"C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_v2/model_batch_{best_batch_size}_exp{num}_k{k}_m{m}.pth"

        X_total, y_total, x, u_min, u_max, u_normalized = pre_data(csv_path, num, k, max(m_values))
        _, _, X_test, y_test = create_dataloaders(X_total, y_total, best_batch_size)

        results_step4 = compare_stage_costs_at_step4(
            model_paths=model_paths,
            test_data=(X_test, y_test),
            k=k,
            m_values=m_values,
            device=device
        )

        plot_step4_stage_comparison(results_step4, save_dir, f"_dataset_{num}")
        
        results_multi = compare_stage_costs_at_multiple_steps(
            model_paths=model_paths,
            test_data=(X_test, y_test),
            k=k,
            m_values=m_values,
            steps_to_predict=[4, 8, 12, 16, 20, 24, 28],
            device=device
        )
        
        plot_multiple_steps_stage_comparison(results_multi, save_dir, f"_dataset_{num}")

    plot_best_losses(best_train_losses, best_batch_sizes, num_list, m_values)
    
    for i, costs_history in enumerate(dataset_costs_history):
        if costs_history: 
            dataset_name = f"dataset_{num_list[i]}"
            plot_layer_wise_performance(costs_history, save_dir, dataset_name)

    plot_multi_dataset_costs(dataset_costs_history, num_list, save_dir)
    plot_concat_layerwise_costs(dataset_costs_history, save_dir)

def predict_main():
    device = torch.device('cuda')
    print(f"Using device: {device}")

    csv_paths = [
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_1.csv",
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_1.csv",
        "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV/mKdV_two_solitons_noise_1.csv"
    ]

    num_list = [15, 30, 45]
    exp_indices = [0, 0, 0]  
    t_start = 5.0
    t_target = 8.0
    t_compare = 7.8
    dt_frame = 0.1
    batch_sizes_list = [
        [32, 64, 128],  
        [32, 64, 128],  
        [32, 64, 128]  
    ]

    k = 64
    m_values = [1, 2, 3, 4]
    m_max = max(m_values)

    for csv_path, num, idx, batch_sizes in zip(csv_paths, num_list, exp_indices, batch_sizes_list):
        print(f"\ndataset: {csv_path} ({num} rogue wave, exp={idx+1})")

        predictions_all_m = {m: {} for m in m_values}
        mse_errors_all_m = {m: {} for m in m_values}
        true_values_all_m = {m: {} for m in m_values}
        times_all_m = {}

        for m in m_values:
            _, _, x, u_min, u_max, u_normalized = pre_data(csv_path, 1, k, m)
            index_start = 70 + idx * 201
            index_target = 101 + idx * 201
            u_initial = u_normalized[index_start-k:index_start] 
            X_compare = u_normalized[index_start:index_target]

            for batch_size in batch_sizes:
                print(f"\npredict batch_size={batch_size}, m={m}")

                model = CNN(k=k, m=m, m_max=m_max).to(device)
                save_path = f"C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_v2/model_batch_{batch_size}_exp{num}_k{k}_m{m}.pth"
                model.load_state_dict(torch.load(save_path))

                start_time = __import__('time').time()
                u_pred_denorm, t_pred, predictions, mse_errors, times, true_values, metrics_records = predict_at_time(
                    model, u_initial, t_target, t_start, dt_frame, device, u_min, u_max, X_compare, k, m
                )
                comp_time_sec = __import__('time').time() - start_time
                predictions_all_m[m][batch_size] = predictions
                mse_errors_all_m[m][batch_size] = mse_errors
                true_values_all_m[m][batch_size] = true_values
                times_all_m[m] = times

                print(f"Prediction completed for batch_size={batch_size}, m={m}, t={t_pred:.2f}")

                                                                    
                save_dir_metrics = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Final code/Final_Final_code/CNNmodels_multifinal_results_v2"
                os.makedirs(save_dir_metrics, exist_ok=True)
                import pandas as pd
                df_metrics = pd.DataFrame(metrics_records)
                df_metrics['batch_size'] = int(batch_size)
                df_metrics['m'] = int(m)
                df_metrics['computational_time_s'] = float(comp_time_sec)
                metrics_csv = os.path.join(save_dir_metrics, f'metrics_summary_CNN_batch{batch_size}_m{m}_exp{num}.csv')
                if os.path.exists(metrics_csv):
                    try:
                        df_old = pd.read_csv(metrics_csv)
                    except Exception:
                        df_old = pd.DataFrame()
                    df_metrics = pd.concat([df_old, df_metrics], ignore_index=True)
                df_metrics.to_csv(metrics_csv, index=False)

                plot_comparison_value_at_time(t_compare, t_start, dt_frame, predictions, true_values, x, batch_size, num, idx + 1, k, m)

            plot_mse(mse_errors_all_m[m], times_all_m[m], num, idx + 1, k, m_values)

        for batch_size in batch_sizes:
            predictions_dict = {m: predictions_all_m[m][batch_size] for m in m_values}
            true_values_dict = {m: true_values_all_m[m][batch_size] for m in m_values}
            plot_comparison_m_values(t_compare, t_start, dt_frame, predictions_dict, true_values_dict, x, batch_size, num, idx + 1, k, m_values)

        for batch_size in batch_sizes:
            mse_errors_dict = {m: mse_errors_all_m[m][batch_size] for m in m_values}
            plot_mse_m_values(mse_errors_dict, times_all_m[m_values[0]], num, idx + 1, batch_size, k, m_values)

if __name__ == '__main__':
    train_main()
    predict_main()