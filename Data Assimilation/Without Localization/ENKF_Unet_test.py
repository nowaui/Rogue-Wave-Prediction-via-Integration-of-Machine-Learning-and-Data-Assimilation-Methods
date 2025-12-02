import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import pandas as pd

np.random.seed(40)
torch.manual_seed(40)

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

             
class UNetModel(nn.Module):
    def __init__(self, k, m, m_max, kernel_size=3):
        super(UNetModel, self).__init__()
        self.kernel_size = kernel_size
        self.pad = (self.kernel_size - 1) // 2
        self.m = m
        self.m_max = m_max
        
                 
        self.enc1_conv1 = nn.Conv1d(k, 16, kernel_size, padding='valid')
        self.enc1_conv2 = nn.Conv1d(16, 16, kernel_size, padding='valid')
        self.pool1 = nn.AdaptiveMaxPool1d(256)
        
        self.enc2_conv1 = nn.Conv1d(16, 32, kernel_size, padding='valid')
        self.enc2_conv2 = nn.Conv1d(32, 32, kernel_size, padding='valid')
        self.pool2 = nn.AdaptiveMaxPool1d(128)
        
        self.enc3_conv1 = nn.Conv1d(32, 64, kernel_size, padding='valid')
        self.enc3_conv2 = nn.Conv1d(64, 64, kernel_size, padding='valid')
        self.pool3 = nn.AdaptiveMaxPool1d(64)
        
        self.bottleneck_conv1 = nn.Conv1d(64, 128, kernel_size, padding='valid')
        self.bottleneck_conv2 = nn.Conv1d(128, 128, kernel_size, padding='valid')
        
                 
        self.up3 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec3_conv1 = nn.Conv1d(128, 64, kernel_size, padding='valid')
        self.dec3_conv2 = nn.Conv1d(64, 64, kernel_size, padding='valid')
        
        self.up2 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec2_conv1 = nn.Conv1d(64, 32, kernel_size, padding='valid')
        self.dec2_conv2 = nn.Conv1d(32, 32, kernel_size, padding='valid')
        
        self.up1 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec1_conv1 = nn.Conv1d(32, 16, kernel_size, padding='valid')
        self.dec1_conv2 = nn.Conv1d(16, 16, kernel_size, padding='valid')
        
        self.final_conv = nn.Conv1d(16, m_max, kernel_size, padding='valid')
        self.relu = nn.ReLU()

    def forward(self, x):
                                 
        x = F.pad(x, (self.pad, self.pad), mode='circular')                        

        e1 = self.enc1_conv1(x)                         
        e1 = self.relu(e1)
        e1 = F.pad(e1, (self.pad, self.pad), mode='circular')
        e1 = self.enc1_conv2(e1)
        e1 = self.relu(e1)
        e1 = F.pad(e1, (self.pad, self.pad), mode='circular')
        p1 = self.pool1(e1)                         

        p1 = F.pad(p1, (self.pad, self.pad), mode='circular')
        e2 = self.enc2_conv1(p1)
        e2 = self.relu(e2)
        e2 = F.pad(e2, (self.pad, self.pad), mode='circular')
        e2 = self.enc2_conv2(e2)
        e2 = self.relu(e2)
        e2 = F.pad(e2, (self.pad, self.pad), mode='circular')
        p2 = self.pool2(e2)                         

        p2 = F.pad(p2, (self.pad, self.pad), mode='circular')
        e3 = self.enc3_conv1(p2)
        e3 = self.relu(e3)
        e3 = F.pad(e3, (self.pad, self.pad), mode='circular')
        e3 = self.enc3_conv2(e3)
        e3 = self.relu(e3)
        e3 = F.pad(e3, (self.pad, self.pad), mode='circular')
        p3 = self.pool3(e3)                        

        p3 = F.pad(p3, (self.pad, self.pad), mode='circular')
        b = self.bottleneck_conv1(p3)
        b = self.relu(b)
        b = F.pad(b, (self.pad, self.pad), mode='circular')
        b = self.bottleneck_conv2(b)
        b = self.relu(b)
        b = F.pad(b, (self.pad, self.pad), mode='circular')

                 
        d3 = self.up3(b)
        d3 = F.pad(d3, (self.pad, self.pad), mode='circular')
        d3 = torch.cat([e3[:, :, 1:-1], d3[:, :, 3:-3]], dim=1)
        d3 = F.pad(d3, (self.pad, self.pad), mode='circular')
        d3 = self.dec3_conv1(d3)
        d3 = self.relu(d3)
        d3 = F.pad(d3, (self.pad, self.pad), mode='circular')
        d3 = self.dec3_conv2(d3)
        d3 = self.relu(d3)
        d3 = F.pad(d3, (self.pad, self.pad), mode='circular')

        d2 = self.up2(d3)
        d2 = F.pad(d2, (self.pad, self.pad), mode='circular')
        d2 = torch.cat([e2[:, :, 1:-1], d2[:, :, 3:-3]], dim=1)
        d2 = F.pad(d2, (self.pad, self.pad), mode='circular')
        d2 = self.dec2_conv1(d2)
        d2 = self.relu(d2)
        d2 = F.pad(d2, (self.pad, self.pad), mode='circular')
        d2 = self.dec2_conv2(d2)
        d2 = self.relu(d2)
        d2 = F.pad(d2, (self.pad, self.pad), mode='circular')

        d1 = self.up1(d2)
        d1 = F.pad(d1, (self.pad, self.pad), mode='circular')
        d1 = d1[:, :, 8:-8]
        d1 = torch.cat([e1, d1], dim=1)
        d1 = F.pad(d1, (self.pad, self.pad), mode='circular')
        d1 = self.dec1_conv1(d1)
        d1 = self.relu(d1)
        d1 = F.pad(d1, (self.pad, self.pad), mode='circular')
        d1 = self.dec1_conv2(d1)
        d1 = self.relu(d1)

                           
        d1 = F.pad(d1, (self.pad, self.pad), mode='circular')
        out = self.final_conv(d1)
        out = out[:, :self.m, 1:-1]                        

        return out

class UNetSystemK1:
    def __init__(self, unet_model, device, m=1, u_min=-1.0, u_max=1.0, dx=0.1, dt_frame=0.1, nu=0.0):
        self.unet_model = unet_model
        self.device = device
        self.m = m
        self.M = 500
        self.u_min = u_min
        self.u_max = u_max
        self.dx = dx
        self.dt_frame = dt_frame
        self.nu = nu
    
    def normalize(self, u):
        return 2 * (u - self.u_min) / (self.u_max - self.u_min) - 1
    
    def denormalize(self, u_norm):
        return (u_norm + 1) * (self.u_max - self.u_min) / 2 + self.u_min
    
    def predict_ensemble(self, u_ensemble, skip_steps):
        if u_ensemble.ndim == 1:
            u_ensemble = u_ensemble.reshape(-1, 1)
            is_1d = True
        else:
            is_1d = False
        
        u_current = u_ensemble.copy()
        u_current_norm = self.normalize(u_current)
        states = [u_current.copy()]
        
        u_input = torch.tensor(u_current_norm.T, device=self.device, dtype=torch.float32)
        model_input = u_input.unsqueeze(1)
        
        self.unet_model.eval()
        with torch.no_grad():
            delta_pred = self.unet_model(model_input)
            delta_pred = delta_pred.cpu().numpy()
        
        for step in range(min(skip_steps, self.m)):
            step_delta = delta_pred[:, step, :].T
            u_current_norm += step_delta
            u_current = self.denormalize(u_current_norm)

            if self.nu != 0.0:
                diffusion_term = (np.roll(u_current, -1, axis=0) - 2 * u_current + np.roll(u_current, 1, axis=0)) / (self.dx ** 2)
                u_current = u_current + self.nu * self.dt_frame * diffusion_term

            states.append(u_current.copy())

            if step < skip_steps - 1:
                u_current_norm = self.normalize(u_current)
        
                                                            
        return states

class mKdVSystem:    
    def __init__(self, M, dx, dt, epsilon=6.0, mu=1.0, nu=0.01):
        self.M = M
        self.dx = dx
        self.dt = dt
        self.epsilon = epsilon
        self.mu = mu
        self.nu = nu
        self.inv_2dx = 1.0 / (2.0 * dx)
        self.inv_2dx3 = 1.0 / (2.0 * dx**3)
        self.inv_dx2 = 1.0 / (dx**2)
    
    def F(self, xi_ensemble):
        if xi_ensemble.ndim == 1:
            xi_ensemble = xi_ensemble.reshape(-1, 1)
            is_1d = True
        else:
            is_1d = False

        M, N = xi_ensemble.shape
        F_ensemble = np.zeros((M, N))
        
                                      
        xi_ext = np.zeros((M + 4, N))
        xi_ext[2:M+2, :] = xi_ensemble
        xi_ext[0:2, :] = xi_ensemble[M-2:M, :]
        xi_ext[M+2:M+4, :] = xi_ensemble[0:2, :]
        
        for m in range(M):
            m_ext = m + 2
            nonlinear = -self.epsilon * xi_ext[m_ext, :]**2 * (xi_ext[m_ext+1, :] - xi_ext[m_ext-1, :]) * self.inv_2dx
            linear = -self.mu * (xi_ext[m_ext+2, :] - 2*xi_ext[m_ext+1, :] + 2*xi_ext[m_ext-1, :] - xi_ext[m_ext-2, :]) * self.inv_2dx3
            diffusion = self.nu * (xi_ext[m_ext+1, :] - 2*xi_ext[m_ext, :] + xi_ext[m_ext-1, :]) * self.inv_dx2
            F_ensemble[m, :] = nonlinear + linear + diffusion
        
        return F_ensemble.flatten() if is_1d else F_ensemble
    
    def M_op(self, xi_ensemble):
        k1 = self.F(xi_ensemble)
        k2 = self.F(xi_ensemble + 0.5 * self.dt * k1)
        k3 = self.F(xi_ensemble + 0.5 * self.dt * k2)
        k4 = self.F(xi_ensemble + self.dt * k3)
        xi_new = xi_ensemble + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return xi_new

class EnKF_UNet:   
    def __init__(self, unet_system, N, Q_std, R_std, Q_d):
        self.unet_system = unet_system
        self.M_dim = unet_system.M
        self.N = N
        self.Q_std = Q_std
        self.R_std = R_std
        
        self.Q = self.create_gaussian_Q(Q_d)
        self.L_Q = np.linalg.cholesky(self.Q)
        self.G = np.eye(self.M_dim)
    
    def create_gaussian_Q(self, Q_d):
        Q = np.zeros((self.M_dim, self.M_dim))
        for i in range(self.M_dim):
            for j in range(self.M_dim):
                dist = min(abs(i - j), self.M_dim - abs(i - j))
                Q[i, j] = self.Q_std**2 * np.exp(-dist**2 / (2 * Q_d**2))
        Q = (Q + Q.T) / 2
        Q += 1e-10 * np.eye(self.M_dim)
        return Q
    
    def generate_process_noise(self):
        eta = np.random.normal(0, 1, (self.M_dim, self.N))
        return self.L_Q @ eta
    
    def initialize_enkf(self, x0, P0_std=0.1):
        xi_ensemble = np.zeros((self.M_dim, self.N))
        
        perturbation = np.random.normal(0, P0_std, (self.M_dim, self.N))
        xi_ensemble= x0[:, np.newaxis] + perturbation
        
        return xi_ensemble
    
    def time_update(self, xi_a):
        
        w_k = self.generate_process_noise()
        xi_f = self.unet_system.predict_ensemble(xi_a) + self.G @ w_k
        return xi_f
    
    def compute_mean_anomaly(self, xi_f):
              
        x_f = np.mean(xi_f, axis=1)
        
                            
        xi_anomaly = xi_f - x_f[:, np.newaxis]
        L_f = xi_anomaly / np.sqrt(self.N - 1)
        return x_f, L_f
    
    def measurement_update(self, xi_f, z_k, H, R):
        x_f, L_f = self.compute_mean_anomaly(xi_f)
        
                           
        Psi_k = H @ L_f
        
                     
        innovation_cov = Psi_k @ Psi_k.T + R

        K_k = L_f @ Psi_k.T @ np.linalg.inv(innovation_cov)

        
        v_k = self.R_std * np.random.randn(len(z_k), self.N)
        H_xi_f = H @ xi_f
        innovations = z_k[:, np.newaxis] - H_xi_f - v_k
        xi_a = xi_f + K_k @ innovations
        return xi_a, K_k
    
    def get_mean_and_spread(self, xi):
        mean = np.mean(xi, axis=1)
        stand_dev = np.std(xi, axis=1)
        return mean, stand_dev

def two_soliton_ic(x, a=[1.0, -0.8], x0=[0.0, 0.0], t0=-5.0):
    u0 = np.zeros_like(x)
    for i in range(len(a)):
        u0 += a[i] / np.cosh(a[i] * (x - x0[i] - a[i]**3 * t0))
    return u0

def get_model_path(exp=30, batch_size=64, m=1):
    base_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Unetmodels_multifinal_v2"
    filename = f"model_batch_{batch_size}_exp{exp}_k1_m{m}.pth"
    return os.path.join(base_dir, filename)

def run_enkf_unet_experiment(target_time, N_ensemble, Q_d):
                       
    L = 50.0
    dx = 0.1
    M = int(L / dx)
    dt_numerical = 1e-4
    dt_frame = 0.1
    
    T_numerical = target_time + 0.1
    N_t_numerical = int(T_numerical / dt_numerical)
    N_t_frame = int(target_time / dt_frame)
    
    x = np.linspace(-L/2, L/2, M, endpoint=False)
    target_idx_numerical = int(target_time / dt_numerical)
    
                     
    device = torch.device('cpu')
    m = 4
    model_path = get_model_path(exp=30, batch_size=64, m=m)
    
    print(f"Running EnKF-UNet: target={target_time:.1f}s, N={N_ensemble}, Q_d={Q_d}")
    print(f"UNet time step: {dt_frame:.1f}s, UNet update interval: {m * dt_frame:.1f}s")
    print(f"Numerical time step: {dt_numerical:.1e}s")
    
    unet_model = UNetModel(k=1, m=m, m_max=4, kernel_size=3)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.to(device)
    unet_model.eval()
    
                         
    truth_T_available = T_numerical
    store_filename = f"u_truth_T{truth_T_available}_dt{dt_numerical}_diffusion_long.csv"
    store_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\truth_data_new"
    os.makedirs(store_dir, exist_ok=True)
    store_path = os.path.join(store_dir, store_filename)
    u0_truth = None
    
    if os.path.exists(store_path):
        print(f"loading: {store_filename}")
        u_truth = pd.read_csv(store_path, header=None).values
        required_rows = N_t_numerical + 1
        if u_truth.shape[1] == M and u_truth.shape[0] >= required_rows:
            u_truth = u_truth[:required_rows, :]
            u0_truth = u_truth[0].copy()
        else:
            u_truth = None
    else:
        u_truth = None
    
    if u_truth is None:
        print("Generating truth solution")
        start_time = time.time()
        mkdv_system_numerical = mKdVSystem(M, dx, dt_numerical)
        u0_truth = two_soliton_ic(x)
        u_truth = np.zeros((N_t_numerical + 1, M))
        u_truth[0] = u0_truth
        
        u = u0_truth.copy()
        for k in range(N_t_numerical):
            if (k+1) % 10000 == 0:
                progress = (k+1)/N_t_numerical*100
                elapsed = time.time() - start_time
                eta = elapsed / (k+1) * (N_t_numerical - k - 1)
                print(f"  Progress: {progress:.1f}%, ETA: {eta:.1f}s")
                if k % 10000 == 0:
                    print(f"Saved {k+1} time points")
            u = mkdv_system_numerical.M_op(u)
            u_truth[k+1] = u
        
        generation_time = time.time() - start_time
        print(f"Generated truth solution in {generation_time:.2f}s")
        
                              
        pd.DataFrame(u_truth).to_csv(store_path, index=False, header=False, float_format='%.8e')
   
    u_min, u_max = np.min(u_truth), np.max(u_truth)
    
                        
    obs_ratio = 0.1
    obs_interval = 3 

    ratio = int(dt_frame / dt_numerical)
    sqrt_ratio = np.sqrt(ratio)  
    obs_noise_std = 0.1
    
    n_obs = int(obs_ratio * M)
    obs_indices = (np.arange(n_obs) * M // n_obs).astype(int)
    
    H = np.zeros((n_obs, M))
    for i, idx in enumerate(obs_indices):
        H[i, idx] = 1.0
    R = (obs_noise_std**2) * np.eye(n_obs)
    
                           
    observations = {}
    for k in range(0, N_t_frame + 1, obs_interval):
        numerical_idx = int(k * dt_frame / dt_numerical)
        if numerical_idx < len(u_truth):
            y_true = u_truth[numerical_idx, obs_indices]
            noise = obs_noise_std * np.random.normal(0, 1, n_obs)
            observations[k] = y_true + noise
    
    print(f"Generated {len(observations)} observation points")
    
                     
    unet_system = UNetSystemK1(unet_model, device, m=m, u_min=u_min, u_max=u_max, dx=dx, dt_frame=dt_frame, nu=0.01)
    enkf_unet = EnKF_UNet(unet_system, N_ensemble, Q_std= sqrt_ratio * 5e-4, R_std=obs_noise_std, Q_d=Q_d)
    
                         
    x0_est = np.zeros_like(u0_truth)
                             
    xi_ensemble = enkf_unet.initialize_enkf(x0_est, P0_std=0.1)
    
                        
    x_mean_history = np.zeros((N_t_numerical + 1, M))
    std_history = np.zeros((N_t_numerical + 1, M))
    mse_history = np.zeros(N_t_numerical + 1)
    residual_mean_history = np.zeros(N_t_numerical + 1)
    spread_history = np.zeros(N_t_numerical + 1)

    filled_indices = np.zeros(N_t_numerical + 1, dtype=bool)
    
    x_mean, std = enkf_unet.get_mean_and_spread(xi_ensemble)
    x_mean_history[0] = x_mean
    std_history[0] = std
    mse_history[0] = np.mean((x_mean - u_truth[0])**2)
    residual_mean_history[0] = np.mean(u_truth[0] - x_mean)
    spread_history[0] = np.mean(std)
    filled_indices[0] = True
    
    if 0 in observations and n_obs > 0:
        z_0 = observations[0]
        xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_0, H, R)
        x_mean, std = enkf_unet.get_mean_and_spread(xi_ensemble)
        x_mean_history[0] = x_mean
        std_history[0] = std
        mse_history[0] = np.mean((x_mean - u_truth[0])**2)
        residual_mean_history[0] = np.mean(u_truth[0] - x_mean)
        spread_history[0] = np.mean(std)
    
    start_time = time.time()
    for k in range(0, N_t_frame, m):
        if k % 10 == 0:
            progress = k / N_t_frame * 100
            current_numerical_idx = int(k * dt_frame / dt_numerical)
            if current_numerical_idx < len(mse_history):
                print(f"  Progress: {progress:.0f}%, MSE: {mse_history[current_numerical_idx]:.6f}")

        states = unet_system.predict_ensemble(xi_ensemble, skip_steps=m)

        for step in range(m):
            frame_idx = k + step + 1
            if frame_idx > N_t_frame:
                break

            xi_pred = unet_system.predict_ensemble(xi_ensemble, skip_steps=1)[1]
            xi_ensemble = xi_pred + enkf_unet.G @ enkf_unet.generate_process_noise()

            z_k = observations.get(frame_idx)
            if z_k is not None:
                xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_k, H, R)

            current_time = frame_idx * dt_frame
            start_numerical_idx = int((current_time - dt_frame) / dt_numerical) + 1
            end_numerical_idx   = int(current_time / dt_numerical) + 1

            x_mean, std = enkf_unet.get_mean_and_spread(xi_ensemble)
            for num_idx in range(start_numerical_idx, min(end_numerical_idx + 1, len(x_mean_history))):
                x_mean_history[num_idx] = x_mean
                std_history[num_idx] = std
                mse_history[num_idx] = np.mean((x_mean - u_truth[num_idx])**2)
                residual_mean_history[num_idx] = np.mean(u_truth[num_idx] - x_mean)
                spread_history[num_idx] = np.mean(std)
                filled_indices[num_idx] = True

    
                                      
                         
                                            
                                                                      
                                                          
                                                                                                      
        
                       
                                                  
                                                                                                   

        
                               
                                      
                                        
                                       
                                                   
                                     
                                                                                           
                
                                  
                                                     
                                                                                         
                                                                          
                
                                                                          
                                                                                                             
                                                      
                                                
                                                                                    
                                                                                         
                                                            
    
    running_time = time.time() - start_time
    print(f"Completed in {running_time:.2f}s")
    print(f"Filled {np.sum(filled_indices)} out of {len(filled_indices)} time points")
    
    return {
        'u_truth': u_truth,
        'x_mean': x_mean_history,
        'std': std_history,
        'mse': mse_history,
        'residual_mean': residual_mean_history,
        'spread': spread_history,
        'x': x,
        'target_idx_numerical': target_idx_numerical,
        'running_time': running_time,
        'dt_frame': dt_frame,
        'dt_numerical': dt_numerical,
        'filled_indices': filled_indices
    }

def create_all_figures_unet(target_time, N_ensemble, Q_d, time_chosen=None):
    save_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_Unet_results"
    os.makedirs(save_dir, exist_ok=True)
    
    results = run_enkf_unet_experiment(target_time, N_ensemble, Q_d)
    
    u_truth = results['u_truth']
    x_mean = results['x_mean']
    std = results['std']
    mse = results['mse']
    residual_mean = results['residual_mean']
    spread = results['spread']
    x = results['x']
    target_idx_numerical = results['target_idx_numerical']
    dt_frame = results['dt_frame']
    dt_numerical = results['dt_numerical']
    running_time = results['running_time']
    
    time_vec = np.arange(len(mse)) * dt_numerical

                                                                                
    selected_time_s = float(time_chosen) if time_chosen is not None else float(target_time)
                                                 
    selected_idx_numerical = int(round(selected_time_s / dt_numerical))
    selected_idx_numerical = max(0, min(selected_idx_numerical, len(u_truth) - 1))
                                        
    selected_truth = u_truth[selected_idx_numerical]
    selected_estimate = x_mean[selected_idx_numerical]
    selected_std = std[selected_idx_numerical]
                                                                    
    sample_step = 1000
    mask_upto = time_vec <= (selected_time_s + 1e-12)
    t_plot = time_vec[mask_upto][::sample_step]
    mse_plot = mse[mask_upto][::sample_step]
    residual_plot = residual_mean[mask_upto][::sample_step]
    spread_plot = spread[mask_upto][::sample_step]
    
                           
    plt.figure(figsize=(12, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=2.5, label='Truth')
    plt.plot(x, selected_estimate, 'r--', linewidth=2.5, label='EnKF-UNet')
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF-UNet prediction (assimilation up to t = {target_time:.2f})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_vs_truth_t{selected_time_s:.2f}s_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
    plt.close()
    
                              
    plt.figure(figsize=(12, 6))
    plt.semilogy(t_plot, mse_plot, 'g-', linewidth=2.5, label='MSE')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, 
                label=f'Target time (t={selected_time_s:.2f}s)')
    plt.xlabel('Time t (s)')
    plt.xlim(0.0, selected_time_s)
    plt.ylim(0.001, 0.1)
    plt.ylabel('MSE')
    plt.title('MSE Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    sampled_mse_full = mse[mask_upto]
    valid_mse_mask = np.isfinite(sampled_mse_full)
    if np.any(valid_mse_mask):
        rmse_final = float(np.sqrt(sampled_mse_full[valid_mse_mask][-1]))
        ax = plt.gca()
        ax.text(0.98, 0.92, f'Final RMSE = {rmse_final:.6f}', transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mse_evolution_t{selected_time_s:.2f}s_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
    plt.show()
    
                                   
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, residual_plot, 'b-', linewidth=2.5, label='Mean Residual')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, label=f'Target t={selected_time_s:.1f}s')
    plt.xlabel('Time t (s)')
    plt.ylabel('Mean Residual')
    plt.title('Mean Residual Evolution over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'residual_evolution_t{selected_time_s:.2f}s_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
    plt.show()
    
                                          
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, spread_plot, 'purple', linewidth=2.5, label='Spread')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, label=f'Target t={selected_time_s:.1f}s')
    plt.xlabel('Time t (s)')
    plt.ylabel('Spread')
    plt.title('Spread Evolution over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    sampled_spread_full = spread[mask_upto]
    valid_spread_mask = np.isfinite(sampled_spread_full)
    if np.any(valid_spread_mask):
        spread_final = float(sampled_spread_full[valid_spread_mask][-1])
        ax = plt.gca()
        ax.text(0.98, 0.92, f'Final Spread = {spread_final:.6f}', transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spread_evolution_t{selected_time_s:.2f}s_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
    plt.show()
    
                 
    plt.figure(figsize=(12, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=3, label='Truth')
    plt.plot(x, selected_estimate, 'r--', linewidth=2.5, label='EnKF-UNet')
    
    lower_2sigma = selected_estimate - 2*selected_std
    upper_2sigma = selected_estimate + 2*selected_std
    plt.fill_between(x, lower_2sigma, upper_2sigma, alpha=0.3, color='red', label='2σ bounds')
    
    plt.xlabel('Spatial Position x')
    plt.ylabel('Wave Amplitude u')
    plt.title(f'EnKF-UNet with Error Bands at t={selected_time_s:.1f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'error_bands_t{selected_time_s:.1f}s_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
    plt.show()

    rmse_at_target = compute_rmse_over_domain(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical])

    rogue_threshold = 1.0
    rogue_region = u_truth[selected_idx_numerical] > rogue_threshold
    if not np.any(rogue_region):
        rogue_region = None
    peak_amp_error = compute_peak_amplitude_error(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical], rogue_region)
    mean_abs_err_peak_region = compute_mean_abs_error_region(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical], rogue_region)

    comp_time_sec = float(running_time)

    print(f"RMSE (domain) at t={selected_time_s:.2f}s: {rmse_at_target:.6f}")
    if rogue_region is None:
        print(f"no region detected")
    else:
        print(f"Mean absolute error in rogue region: {mean_abs_err_peak_region:.6f}")
    print(f"Computational time (s): {comp_time_sec:.2f}")

                                    
    mse_evolution_csv = os.path.join(save_dir, f'mse_evolution_N_{N_ensemble}_Qd_{Q_d}_t_{selected_time_s:.2f}s_Unet.csv')
    mse_data = pd.DataFrame({
        'time': time_vec[mask_upto],
        'mse': mse[mask_upto],
        'residual_mean': residual_mean[mask_upto],
        'spread': spread[mask_upto]
    })
    mse_data.to_csv(mse_evolution_csv, index=False)
    print(f"MSE evolution data saved to: {mse_evolution_csv}")
    
                          
    metrics_csv = os.path.join(save_dir, 'metrics_summary_Unet.csv')
    metrics_row = {
        'target_time_s': float(selected_time_s),
        'run_target_time_s': float(target_time),
        'N_ensemble': int(N_ensemble),
        'Q_d': float(Q_d),
        'rmse_domain': rmse_at_target,
        'peak_amplitude_error': peak_amp_error,
        'computational_time_s': comp_time_sec,
        'mean_abs_error_peak_region': mean_abs_err_peak_region,
        'mse_evolution_file': f'mse_evolution_N_{N_ensemble}_Qd_{Q_d}_t_{selected_time_s:.2f}s_Unet.csv'
    }
    if os.path.exists(metrics_csv):
        try:
            df_metrics = pd.read_csv(metrics_csv)
        except Exception:
            df_metrics = pd.DataFrame()
        df_metrics = pd.concat([df_metrics, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        df_metrics = pd.DataFrame([metrics_row])
    df_metrics.to_csv(metrics_csv, index=False)

    print(f"Results saved to {save_dir}")
    print(f"Final MSE: {mse[-1]:.6f}")
    print(f"Final Mean Residual: {residual_mean[-1]:.6f}")
    print(f"Final Ensemble Spread: {spread[-1]:.6f}")

if __name__ == "__main__":        
    create_all_figures_unet(target_time=7.8, N_ensemble=150, Q_d=10, time_chosen=7.8)

