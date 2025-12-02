import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import pandas as pd
import random

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
    
    def time_update(self, xi_a, skip_steps=1, add_noise=True):
        xi_pred = self.unet_system.predict_ensemble(xi_a, skip_steps=skip_steps)[skip_steps]
        if add_noise:
            return xi_pred + self.G @ self.generate_process_noise()
        return xi_pred
    
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

def get_model_path(exp=45, batch_size=32, m=4):
    base_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Unetmodels_multifinal_v2"
    filename = f"model_batch_{batch_size}_exp{exp}_k1_m{m}.pth"
    return os.path.join(base_dir, filename)

def adjust_ensemble_member(x_T, tau_i, alpha):
    return x_T + alpha * (tau_i - x_T)

def run_enkf_enhanced_unet_experiment(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                                      selected_members=None, alpha=1.0, n_prediction_ensemble=None, 
                                      save_data=True):
    if n_prediction_ensemble is None:
        n_prediction_ensemble = N_ensemble
    
    base_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_Unet_newresults"
    os.makedirs(base_dir, exist_ok=True)
    
                                                                             
    exp_name = (
        f"Results_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}_"
        f"lead{lead_time:.1f}_alpha{alpha}_Npred{n_prediction_ensemble}"
    )
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    analysis_dir = os.path.join(exp_dir, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
                                         
    analysis_files = {
        'mean_history': os.path.join(analysis_dir, 'x_mean_history.csv'),
        'std_history': os.path.join(analysis_dir, 'std_history.csv'),
        'mse_history': os.path.join(analysis_dir, 'mse_history.csv'),
        'residual_history': os.path.join(analysis_dir, 'residual_mean_history.csv'),
        'spread_history': os.path.join(analysis_dir, 'spread_history.csv'),
        'final_ensemble': os.path.join(analysis_dir, 'final_ensemble.csv'),
        'parameters': os.path.join(analysis_dir, 'parameters.csv')
    }
    
                                                     
    load_analysis = False
    if save_data and all(os.path.exists(f) for f in analysis_files.values()):
        try:
            params_df = pd.read_csv(analysis_files['parameters'])
            stored_params = params_df.iloc[0]
            
            if (stored_params['target_time'] == target_time and
                stored_params['analysis_time'] == analysis_time and
                stored_params['N_ensemble'] == N_ensemble and
                stored_params['Q_d'] == Q_d):
                load_analysis = True
                print("Loading existing analysis data...")
        except Exception as e:
            print(f"Error reading analysis parameters: {e}")
            load_analysis = False
    
    if load_analysis:
                                  
        try:
            x_mean_analysis = pd.read_csv(analysis_files['mean_history'], header=None).values
            std_analysis = pd.read_csv(analysis_files['std_history'], header=None).values
            mse_analysis = pd.read_csv(analysis_files['mse_history'], header=None).values.flatten()
            residual_analysis = pd.read_csv(analysis_files['residual_history'], header=None).values.flatten()
            spread_analysis = pd.read_csv(analysis_files['spread_history'], header=None).values.flatten()
            xi_ensemble = pd.read_csv(analysis_files['final_ensemble'], header=None).values
            
                              
            M = xi_ensemble.shape[0]
            if xi_ensemble.shape[1] != N_ensemble:
                raise ValueError(f"Ensemble size mismatch: loaded {xi_ensemble.shape[1]}, expected {N_ensemble}")
                
            print(f"Successfully loaded analysis data with M={M}, N_ensemble={N_ensemble}")
            
                                                
            print(f"  Loaded xi_ensemble range: [{np.min(xi_ensemble):.6f}, {np.max(xi_ensemble):.6f}]")
            print(f"  Loaded ensemble mean: {np.mean(xi_ensemble):.6f}")
            
                                                  
            L = 50.0
            dx = 0.1
            dt_numerical = 1e-4
            dt_frame = 0.1
            
            T_numerical = target_time + 0.1
            N_t_numerical = int(T_numerical / dt_numerical)
            
            x = np.linspace(-L/2, L/2, M, endpoint=False)
            
            truth_T_available = T_numerical
            store_filename = f"u_truth_T{truth_T_available}_dt{dt_numerical}_diffusion_long.csv"
            store_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\truth_data_new"
            store_path = os.path.join(store_dir, store_filename)
            
            if os.path.exists(store_path):
                u_truth = pd.read_csv(store_path, header=None).values
                required_rows = N_t_numerical + 1
                if u_truth.shape[1] == M and u_truth.shape[0] >= required_rows:
                    u_truth = u_truth[:required_rows, :]
                else:
                    raise ValueError("Truth data dimensions don't match")
            else:
                raise ValueError("Truth data file not found")
                
        except Exception as e:
            print(f"Error loading analysis data: {e}")
            print("Regenerating analysis phase...")
            load_analysis = False
    
    if not load_analysis:
                                   
        print("Phase 1: Data Assimilation - Computing new analysis")
        
                           
        L = 50.0
        dx = 0.1
        M = int(L / dx)
        dt_numerical = 1e-4
        dt_frame = 0.1
        
        T_numerical = target_time + 0.1
        N_t_numerical = int(T_numerical / dt_numerical)
        analysis_idx = int(analysis_time / dt_numerical)
        
        x = np.linspace(-L/2, L/2, M, endpoint=False)
        
                                         
        truth_T_available = T_numerical
        store_filename = f"u_truth_T{truth_T_available}_dt{dt_numerical}_diffusion_long.csv"
        store_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\truth_data_new"
        store_path = os.path.join(store_dir, store_filename)
        
        if os.path.exists(store_path):
            print(f"Loading truth solution: {store_filename}")
            u_truth = pd.read_csv(store_path, header=None).values
            required_rows = N_t_numerical + 1
            if u_truth.shape[1] == M and u_truth.shape[0] >= required_rows:
                u_truth = u_truth[:required_rows, :]
            else:
                raise ValueError("Truth data dimensions don't match")
        else:
            raise ValueError("Truth data file not found")
        
                                                     
        obs_ratio = 0.1
        obs_interval = 3
        obs_noise_std = 0.1
        
        n_obs = int(obs_ratio * M)
        obs_indices = (np.arange(n_obs) * M // n_obs).astype(int)  
                                                                          
        
        H = np.zeros((n_obs, M))
        for i, idx in enumerate(obs_indices):
            H[i, idx] = 1.0
        R = (obs_noise_std**2) * np.eye(n_obs)
        
                                                        
        observations = {}
        N_t_frame = int(target_time / dt_frame)
        for k in range(0, N_t_frame + 1, obs_interval):
            numerical_idx = int(k * dt_frame / dt_numerical)
            if numerical_idx <= analysis_idx and numerical_idx < len(u_truth):
                y_true = u_truth[numerical_idx, obs_indices]
                noise = obs_noise_std * np.random.normal(0, 1, n_obs)
                observations[k] = y_true + noise
        
                                
        device = torch.device('cpu')
        m = 4
        model_path = get_model_path(exp=45, batch_size=32, m=m)
        
        unet_model = UNetModel(k=1, m=m, m_max=4, kernel_size=3)
        unet_model.load_state_dict(torch.load(model_path, map_location=device))
        unet_model.to(device)
        unet_model.eval()
        
        u_min, u_max = np.min(u_truth), np.max(u_truth)
        ratio = int(dt_frame / dt_numerical)
        sqrt_ratio = np.sqrt(ratio)
        
        unet_system = UNetSystemK1(unet_model, device, m=m, u_min=u_min, u_max=u_max, dx=dx, dt_frame=dt_frame, nu=0.01)
        enkf_unet = EnKF_UNet(unet_system, N_ensemble, Q_std=sqrt_ratio * 5e-4, R_std=obs_noise_std, Q_d=Q_d)
        
                             
        x0_est = np.zeros_like(u_truth[0])
        xi_ensemble = enkf_unet.initialize_enkf(x0_est, P0_std=0.1)
        
                                                              
        x_mean_analysis = np.zeros((analysis_idx + 1, M), dtype=np.float32)
        std_analysis = np.zeros((analysis_idx + 1, M), dtype=np.float32)
        mse_analysis = np.zeros(analysis_idx + 1, dtype=np.float32)
        residual_analysis = np.zeros(analysis_idx + 1, dtype=np.float32)
        spread_analysis = np.zeros(analysis_idx + 1, dtype=np.float32)
        
                       
        x_mean, std = enkf_unet.get_mean_and_spread(xi_ensemble)
        x_mean_analysis[0] = x_mean
        std_analysis[0] = std
        mse_analysis[0] = np.mean((x_mean - u_truth[0])**2)
        residual_analysis[0] = np.mean(u_truth[0] - x_mean)
        spread_analysis[0] = np.mean(std)
        
                                                
        if 0 in observations:
            z_0 = observations[0]
            xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_0, H, R)
            x_mean, std = enkf_unet.get_mean_and_spread(xi_ensemble)
            x_mean_analysis[0] = x_mean
            std_analysis[0] = std
            mse_analysis[0] = np.mean((x_mean - u_truth[0])**2)
            residual_analysis[0] = np.mean(u_truth[0] - x_mean)
            spread_analysis[0] = np.mean(std)
        
                      
        da_start_time = time.time()
        N_t_frame_analysis = int(analysis_time / dt_frame)
        
        for k in range(0, N_t_frame_analysis, m):
            if k % 10 == 0:
                progress = k / N_t_frame_analysis * 100
                current_numerical_idx = int(k * dt_frame / dt_numerical)
                if current_numerical_idx < len(mse_analysis):
                    print(f"  DA Progress: {progress:.0f}%, MSE: {mse_analysis[current_numerical_idx]:.6f}")
            
                                              
            for step in range(m):
                frame_idx = k + step + 1
                if frame_idx > N_t_frame_analysis:
                    break
                
                                                                        
                xi_ensemble = enkf_unet.time_update(xi_ensemble, skip_steps=1, add_noise=True)
                
                                                             
                z_k = observations.get(frame_idx)
                if z_k is not None:
                    xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_k, H, R)
                
                                
                current_time = frame_idx * dt_frame
                start_numerical_idx = int((current_time - dt_frame) / dt_numerical) + 1
                end_numerical_idx = int(current_time / dt_numerical) + 1
                
                x_mean, std = enkf_unet.get_mean_and_spread(xi_ensemble)
                for num_idx in range(start_numerical_idx, min(end_numerical_idx + 1, len(x_mean_analysis))):
                    if num_idx < len(x_mean_analysis):
                        x_mean_analysis[num_idx] = x_mean
                        std_analysis[num_idx] = std
                        mse_analysis[num_idx] = np.mean((x_mean - u_truth[num_idx])**2)
                        residual_analysis[num_idx] = np.mean(u_truth[num_idx] - x_mean)
                        spread_analysis[num_idx] = np.mean(std)
        
        da_time = time.time() - da_start_time
        print(f"Data assimilation completed in {da_time:.2f}")
        
                                     
        x_mean_da, std_da = enkf_unet.get_mean_and_spread(xi_ensemble)
        print(f"  DA final state: mean={np.mean(x_mean_da):.6f}, std={np.mean(std_da):.6f}")
        print(f"  DA xi_ensemble shape: {xi_ensemble.shape}, range: [{np.min(xi_ensemble):.6f}, {np.max(xi_ensemble):.6f}]")
        
                                                                    
        if save_data:
                                               
            pd.DataFrame(x_mean_analysis).to_csv(analysis_files['mean_history'], index=False, header=False, float_format='%.8e')
            pd.DataFrame(std_analysis).to_csv(analysis_files['std_history'], index=False, header=False, float_format='%.8e')
            
                          
            pd.DataFrame(mse_analysis).to_csv(analysis_files['mse_history'], index=False, header=False)
            pd.DataFrame(residual_analysis).to_csv(analysis_files['residual_history'], index=False, header=False)
            pd.DataFrame(spread_analysis).to_csv(analysis_files['spread_history'], index=False, header=False)
            
                                                  
            pd.DataFrame(xi_ensemble).to_csv(analysis_files['final_ensemble'], index=False, header=False, float_format='%.8e')
            
                                                      
            params_data = {
                'target_time': [target_time],
                'analysis_time': [analysis_time],
                'N_ensemble': [N_ensemble],
                'Q_d': [Q_d],
                'M': [M]
            }
            pd.DataFrame(params_data).to_csv(analysis_files['parameters'], index=False)
            print(f"Data assimilation results saved to: {analysis_dir}")
    else:
        print("Phase 1: Data Assimilation - Skipped (loaded from existing data)")
                                                  
        L = 50.0
        dx = 0.1
        dt_numerical = 1e-4
        dt_frame = 0.1
        analysis_idx = int(analysis_time / dt_numerical)
    
                              
    print(f"Running prediction phase from t={analysis_time} to t={lead_time}...")
    
                                            
    if selected_members is not None:
        if len(selected_members) != n_prediction_ensemble:
            raise ValueError(f"selected_members length ({len(selected_members)}) != n_prediction_ensemble ({n_prediction_ensemble})")
        prediction_indices = selected_members
    else:
        prediction_indices = np.arange(min(n_prediction_ensemble, N_ensemble))
    
    xi_prediction = xi_ensemble[:, prediction_indices].copy()
    
                            
    if alpha != 1.0:
        analysis_idx = int(analysis_time / dt_numerical)
        x_T = u_truth[analysis_idx]
        
        print(f"  Applying alpha adjustment: alpha={alpha}")
        print(f"  Before alpha adjustment: xi_prediction range: [{np.min(xi_prediction):.6f}, {np.max(xi_prediction):.6f}]")
        
        for i in range(xi_prediction.shape[1]):
            xi_prediction[:, i] = adjust_ensemble_member(x_T, xi_prediction[:, i], alpha)
            
        print(f"  After alpha adjustment: xi_prediction range: [{np.min(xi_prediction):.6f}, {np.max(xi_prediction):.6f}]")
    else:
        print(f"  No alpha adjustment (alpha={alpha})")
    
                                  
    device = torch.device('cpu')
    m = 4
    model_path = get_model_path(exp=45, batch_size=32, m=m)
    
    unet_model = UNetModel(k=1, m=m, m_max=4, kernel_size=3)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.to(device)
    unet_model.eval()
    
    u_min, u_max = np.min(u_truth), np.max(u_truth)
    dx = 0.1
    
    unet_system = UNetSystemK1(unet_model, device, m=m, u_min=u_min, u_max=u_max, dx=dx, dt_frame=dt_frame, nu=0.01)
    
                                                  
    ratio = int(dt_frame / dt_numerical)
    sqrt_ratio = np.sqrt(ratio)
    obs_noise_std = 0.1
                                                                               
    Q_std_pred = 0.0 if abs(alpha) < 1e-12 else (sqrt_ratio * 5e-4)
    enkf_pred = EnKF_UNet(unet_system, n_prediction_ensemble, Q_std=Q_std_pred, R_std=obs_noise_std, Q_d=Q_d)
    
                                 
    print(f"Phase 2: Pure Prediction from t={analysis_time} to t={lead_time}")
    prediction_start_time = time.time()
    
    analysis_idx = int(analysis_time / dt_numerical)
    lead_idx = int(lead_time / dt_numerical)
    prediction_length = lead_idx - analysis_idx
    
    N_t_frame_pred = int((lead_time - analysis_time) / dt_frame)
    
    x_mean_prediction = np.zeros((prediction_length + 1, M), dtype=np.float32)
    std_prediction = np.zeros((prediction_length + 1, M), dtype=np.float32)
    mse_prediction = np.zeros(prediction_length + 1, dtype=np.float32)
    residual_prediction = np.zeros(prediction_length + 1, dtype=np.float32)
    spread_prediction = np.zeros(prediction_length + 1, dtype=np.float32)
    
                                  
    x_mean, std = enkf_pred.get_mean_and_spread(xi_prediction)
    x_mean_prediction[0] = x_mean
    std_prediction[0] = std
    mse_prediction[0] = np.mean((x_mean - u_truth[analysis_idx])**2)
    residual_prediction[0] = np.mean(u_truth[analysis_idx] - x_mean)
    spread_prediction[0] = np.mean(std)
    
                                           
    print(f"  Initial prediction state: mean={np.mean(x_mean):.6f}, std={np.mean(std):.6f}, MSE={mse_prediction[0]:.6f}")
    print(f"  xi_prediction shape: {xi_prediction.shape}, range: [{np.min(xi_prediction):.6f}, {np.max(xi_prediction):.6f}]")
    
    filled_indices_pred = np.zeros(prediction_length + 1, dtype=bool)
    filled_indices_pred[0] = True
    
    current_frame = 0
    for k in range(0, N_t_frame_pred, m):
        if k % 10 == 0:
            progress = k / N_t_frame_pred * 100
            current_numerical_idx = int(k * dt_frame / dt_numerical)
            if current_numerical_idx < len(mse_prediction):
                print(f"  Prediction Progress: {progress:.0f}%, MSE: {mse_prediction[current_numerical_idx]:.6f}")

                                                         
        states = unet_system.predict_ensemble(xi_prediction, skip_steps=m)

        for step in range(m):
            frame_idx = k + step + 1
            if frame_idx > N_t_frame_pred:
                break

                                                                                   
            add_noise_pred = not (abs(alpha) < 1e-12)
            xi_prediction = enkf_pred.time_update(xi_prediction, skip_steps=1, add_noise=add_noise_pred)
            
                                                                     
            if frame_idx <= 3:
                print(f"    Step {frame_idx}: xi_prediction range: [{np.min(xi_prediction):.6f}, {np.max(xi_prediction):.6f}]")

                                        
            current_time = frame_idx * dt_frame                                  
            start_numerical_idx = int((current_time - dt_frame) / dt_numerical) + 1
            end_numerical_idx = int(current_time / dt_numerical) + 1
            
            x_mean, std = enkf_pred.get_mean_and_spread(xi_prediction)
            
                                       
            for num_idx in range(start_numerical_idx, min(end_numerical_idx + 1, len(x_mean_prediction))):
                if num_idx < len(x_mean_prediction):
                    truth_idx = analysis_idx + num_idx
                    if truth_idx < len(u_truth):
                        x_mean_prediction[num_idx] = x_mean
                        std_prediction[num_idx] = std
                        mse_prediction[num_idx] = np.mean((x_mean - u_truth[truth_idx])**2)
                        residual_prediction[num_idx] = np.mean(u_truth[truth_idx] - x_mean)
                        spread_prediction[num_idx] = np.mean(std)
                        filled_indices_pred[num_idx] = True
    
    prediction_time = time.time() - prediction_start_time
    print(f"Prediction completed in {prediction_time:.2f}")
    
    last_idx = len(x_mean_prediction) - 1
    if last_idx >= 0 and not filled_indices_pred[last_idx]:
        x_mean_last, std_last = enkf_pred.get_mean_and_spread(xi_prediction)
        x_mean_prediction[last_idx] = x_mean_last
        std_prediction[last_idx] = std_last
        truth_idx_last = analysis_idx + last_idx
        if 0 <= truth_idx_last < len(u_truth):
            mse_prediction[last_idx] = np.mean((x_mean_last - u_truth[truth_idx_last])**2)
            residual_prediction[last_idx] = np.mean(u_truth[truth_idx_last] - x_mean_last)
            spread_prediction[last_idx] = np.mean(std_last)
        filled_indices_pred[last_idx] = True
    
                                             
    total_length = len(x_mean_analysis) + len(x_mean_prediction) - 1
    total_mean_history = np.zeros((total_length, M), dtype=np.float32)
    total_std_history = np.zeros((total_length, M), dtype=np.float32)
    total_mse_history = np.zeros(total_length, dtype=np.float32)
    total_residual_history = np.zeros(total_length, dtype=np.float32)
    total_spread_history = np.zeros(total_length, dtype=np.float32)
    
                         
    analysis_len = len(x_mean_analysis)
    total_mean_history[:analysis_len] = x_mean_analysis
    total_std_history[:analysis_len] = std_analysis
    total_mse_history[:analysis_len] = mse_analysis
    total_residual_history[:analysis_len] = residual_analysis
    total_spread_history[:analysis_len] = spread_analysis
    
                           
    pred_len = len(x_mean_prediction) - 1
    if pred_len > 0:
        total_mean_history[analysis_len:analysis_len + pred_len] = x_mean_prediction[1:]
        total_std_history[analysis_len:analysis_len + pred_len] = std_prediction[1:]
        total_mse_history[analysis_len:analysis_len + pred_len] = mse_prediction[1:]
        total_residual_history[analysis_len:analysis_len + pred_len] = residual_prediction[1:]
        total_spread_history[analysis_len:analysis_len + pred_len] = spread_prediction[1:]
    
                                                           
    return {
        'u_truth': u_truth,
        'x_mean_total': total_mean_history,
        'std_total': total_std_history,
        'mse_total': total_mse_history,
        'residual_mean_total': total_residual_history,
        'spread_total': total_spread_history,
        'x_mean_da': x_mean_analysis,
        'std_da': std_analysis,
        'mse_da': mse_analysis,
        'x_mean_pred': x_mean_prediction,
        'std_pred': std_prediction,
        'mse_pred': mse_prediction,
        'x': x,
        'analysis_idx': analysis_idx,
        'lead_idx': int(lead_time / dt_numerical),
        'target_idx': int(lead_time / dt_numerical),
        'ensemble_final_da': xi_ensemble,
        'ensemble_final_pred': xi_prediction,
        'selected_members': selected_members,
        'alpha': alpha,
        'da_time': da_time if not load_analysis else 0,
        'prediction_time': prediction_time,
        'da_save_dir': analysis_dir if save_data else None,
        'exp_dir': exp_dir,
        'dt_numerical': dt_numerical,
        'dt_frame': dt_frame,
        'N_ensemble': N_ensemble,
        'n_prediction_ensemble': n_prediction_ensemble
    }

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
    model_path = get_model_path(exp=45, batch_size=32, m=m)
    
    print(f"Running EnKF-UNet: target={target_time:.1f}, N={N_ensemble}, Q_d={Q_d}")
    print(f"UNet time step: {dt_frame:.1f}, UNet update interval: {m * dt_frame:.1f}")
    print(f"Numerical time step: {dt_numerical:.1e}")
    
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
                print(f"  Progress: {progress:.1f}%, ETA: {eta:.1f}")
                if k % 10000 == 0:
                    print(f"Saved {k+1} time points")
            u = mkdv_system_numerical.M_op(u)
            u_truth[k+1] = u
        
        generation_time = time.time() - start_time
        print(f"Generated truth solution in {generation_time:.2f}")
        
                              
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
    
                        
    x_mean_history = np.zeros((N_t_numerical + 1, M), dtype=np.float32)
    std_history = np.zeros((N_t_numerical + 1, M), dtype=np.float32)
    mse_history = np.zeros(N_t_numerical + 1, dtype=np.float32)
    residual_mean_history = np.zeros(N_t_numerical + 1, dtype=np.float32)
    spread_history = np.zeros(N_t_numerical + 1, dtype=np.float32)

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
    print(f"Completed in {running_time:.2f}")
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

def create_enhanced_unet_figures(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                                selected_members=None, alpha=1.0, n_prediction_ensemble=None, 
                                save_data=True, time_chosen=None):
    results = run_enkf_enhanced_unet_experiment(target_time, analysis_time, lead_time, N_ensemble, Q_d,
                                               selected_members, alpha, n_prediction_ensemble, save_data)
    
    u_truth = results['u_truth']
    x_mean = results['x_mean_total']
    std = results['std_total']
    mse = results['mse_total']
    residual_mean = results['residual_mean_total']
    spread = results['spread_total']
    x = results['x']
    dt_numerical = results['dt_numerical']
    analysis_idx = results['analysis_idx']
    lead_idx = results['lead_idx']
    exp_dir = results['exp_dir']
    N_ensemble_actual = results['N_ensemble']
    n_pred_actual = results['n_prediction_ensemble']
    
    time_vec = np.arange(len(mse)) * dt_numerical
    
                            
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    selected_idx_numerical = int(round(selected_time_s / dt_numerical))
    selected_idx_numerical = max(0, min(selected_idx_numerical, len(u_truth) - 1))

    assimilation_cutoff = 6.0
    forecast_end = 7.8
    assimilation_phase_label = f"Assimilation (t <= {assimilation_cutoff:.1f})"
    forecast_phase_label = f"Forecast ({assimilation_cutoff:.1f}-{forecast_end:.1f})"
    assimilation_forecast_note = f"{assimilation_phase_label}, {forecast_phase_label}"
    if selected_time_s <= assimilation_cutoff:
        current_phase_label = assimilation_phase_label
    elif selected_time_s <= forecast_end:
        current_phase_label = forecast_phase_label
    else:
        current_phase_label = f"Forecast (t > {forecast_end:.1f})"
    
                                                                    
    if selected_idx_numerical <= analysis_idx:
        selected_truth = u_truth[selected_idx_numerical]
        selected_estimate = results['x_mean_da'][selected_idx_numerical]
        selected_std = results['std_da'][selected_idx_numerical]
    else:
        pred_idx = selected_idx_numerical - analysis_idx
        x_mean_pred = results.get('x_mean_pred')
        std_pred = results.get('std_pred')
                                                    
        if x_mean_pred is not None and len(x_mean_pred) > 0:
            pred_idx = max(0, min(pred_idx, len(x_mean_pred) - 1))
                                                                                                            
            row = x_mean_pred[pred_idx]
            if not np.any(row):
                for j in range(pred_idx, -1, -1):
                    if np.any(x_mean_pred[j]):
                        pred_idx = j
                        break
            selected_estimate = x_mean_pred[pred_idx]
            selected_std = std_pred[pred_idx] if std_pred is not None else std[pred_idx]
        else:
                                                                       
            selected_estimate = x_mean[selected_idx_numerical]
            selected_std = std[selected_idx_numerical]
        selected_truth = u_truth[selected_idx_numerical]
    
                                               
    def plot_snapshot(time_s, idx_num, truth_vec, estimate_vec, std_vec,
                      phase_label, note_text, filename_suffix=None):
        sample_step = 1000
        mask_upto = time_vec <= (time_s + 1e-12)
        t_plot = time_vec[mask_upto][::sample_step]
        mse_plot = mse[mask_upto][::sample_step]
        residual_plot = residual_mean[mask_upto][::sample_step]
        spread_plot = spread[mask_upto][::sample_step]

        final_idx = min(idx_num, len(mse) - 1)
        if t_plot.size == 0 or abs(t_plot[-1] - time_s) > 1e-12:
            t_plot = np.append(t_plot, time_s)
            mse_plot = np.append(mse_plot, mse[final_idx])
            residual_plot = np.append(residual_plot, residual_mean[final_idx])
            spread_plot = np.append(spread_plot, spread[final_idx])

        suffix = title_suffix if filename_suffix is None else f"{title_suffix}_{filename_suffix}"

                             
        plt.figure(figsize=(12, 6))
        plt.plot(x, truth_vec, 'b-', linewidth=2.5, label='Truth')
        plt.plot(x, estimate_vec, 'r--', linewidth=2.5, label='EnKF-UNet')
        plt.xlabel('Spatial Position x', fontsize=14)
        plt.ylabel('Wave Amplitude u', fontsize=14)
        plt.title(
            f'EnKF-UNet Prediction vs Truth at t = {time_s:.2f} ({note_text})',
            fontsize=16,
            fontweight='bold'
        )
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'prediction_vs_truth_t{time_s:.2f}_{suffix}.png'), dpi=300)
        plt.close()

                       
        plt.figure(figsize=(12, 6))
        plt.semilogy(t_plot, mse_plot, 'g-', linewidth=2.5, label='MSE')
        plt.axvline(x=analysis_time, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Analysis time (t={analysis_time:.1f})')
        plt.axvline(x=time_s, color='red', linestyle='--', linewidth=2, label=f'Target time (t={time_s:.2f})')
        plt.xlabel('Time t', fontsize=14)
        plt.xlim(0.0, time_s)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
        plt.title(
            f'MSE Evolution ({note_text})',
            fontsize=16,
            fontweight='bold'
        )
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        sampled_mse_full = mse[mask_upto]
        valid_mse_mask = np.isfinite(sampled_mse_full)
        if np.any(valid_mse_mask):
            rmse_final = float(np.sqrt(sampled_mse_full[valid_mse_mask][-1]))
            ax = plt.gca()
            ax.text(0.98, 0.92, f'Final RMSE = {rmse_final:.6f}', transform=ax.transAxes,
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'mse_evolution_t{time_s:.2f}_{suffix}.png'), dpi=300)
        plt.close()

                     
        plt.figure(figsize=(12, 6))
        plt.plot(x, truth_vec, 'b-', linewidth=3, label='Truth')
        plt.plot(x, estimate_vec, 'r--', linewidth=2.5, label='EnKF-UNet')
        lower_2sigma = estimate_vec - 2*std_vec
        upper_2sigma = estimate_vec + 2*std_vec
        plt.fill_between(x, lower_2sigma, upper_2sigma, alpha=0.3, color='red', label='2σ bounds')
        plt.xlabel('Spatial Position x', fontsize=14)
        plt.ylabel('Wave Amplitude u', fontsize=14)
        plt.title(
            f'EnKF-UNet with Error Bands at t = {time_s:.2f} '
            f'(Analysis until t={analysis_time:.1f}; {note_text})',
            fontsize=16,
            fontweight='bold'
        )
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'error_bands_t{time_s:.1f}_{suffix}.png'), dpi=300)
        plt.close()

                          
        plt.figure(figsize=(14, 8))
        plt.plot(x, truth_vec, 'k-', linewidth=3, label='Truth', zorder=5)
        if idx_num <= analysis_idx:
            ensemble_snapshot = results['ensemble_final_da']
        else:
            ensemble_snapshot = results['ensemble_final_pred']
        phase_info = phase_label
        num_members = ensemble_snapshot.shape[1]
        member_colors = plt.cm.viridis(np.linspace(0, 1, num_members))
        for j in range(num_members):
            plt.plot(x, ensemble_snapshot[:, j], color=member_colors[j], linewidth=1.2, alpha=0.6)
        plt.xlabel('Spatial Position x', fontsize=14)
        plt.ylabel('Wave Amplitude u', fontsize=14)
        plt.title(
            f'All Ensemble Members vs Truth at t = {time_s:.2f} ({phase_info})',
            fontsize=16,
            fontweight='bold'
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'ensemble_members_vs_truth_t{time_s:.2f}_{suffix}.png'), dpi=300)
        plt.close()

                          
        plt.figure(figsize=(14, 8))
        plt.plot(x, truth_vec, 'k-', linewidth=3, label='Truth', zorder=10)
        if idx_num <= analysis_idx:
            envelope_snapshot = results['ensemble_final_da']
        else:
            envelope_snapshot = results['ensemble_final_pred']
        phase_info_env = phase_label
        lower_envelope = np.min(envelope_snapshot, axis=1)
        upper_envelope = np.max(envelope_snapshot, axis=1)
        plt.plot(x, estimate_vec, 'r--', linewidth=2.0, label='EnKF-UNet')
        plt.fill_between(x, lower_envelope, upper_envelope, color='tab:blue', alpha=0.2, label='Min–Max envelope')
        plt.xlabel('Spatial Position x', fontsize=14)
        plt.ylabel('Wave Amplitude u', fontsize=14)
        plt.title(
            f'EnKF-UNet with Min–Max Error Bands at t = {time_s:.2f} ({phase_info_env}) '
            f'(Analysis until t={analysis_time:.1f}; {note_text})',
            fontsize=16,
            fontweight='bold'
        )
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'prediction_with_minmax_envelope_t{time_s:.2f}_{suffix}.png'), dpi=300)
        plt.close()

    title_suffix = f"N{N_ensemble_actual}_npred{n_pred_actual}"

                                           
    plot_snapshot(selected_time_s, selected_idx_numerical, selected_truth, selected_estimate,
                  selected_std, current_phase_label, assimilation_forecast_note, filename_suffix=None)

                            
    analysis_truth = u_truth[analysis_idx]
    analysis_estimate = results['x_mean_da'][analysis_idx]
    analysis_std_vec = results['std_da'][analysis_idx]
    plot_snapshot(float(analysis_time), analysis_idx, analysis_truth, analysis_estimate,
                  analysis_std_vec, assimilation_phase_label, assimilation_phase_label, filename_suffix="analysis")
    
                       
    rmse_at_target = compute_rmse_over_domain(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical])
    
    rogue_threshold = 1.0
    rogue_region = u_truth[selected_idx_numerical] > rogue_threshold
    if not np.any(rogue_region):
        rogue_region = None
    peak_amp_error = compute_peak_amplitude_error(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical], rogue_region)
    mean_abs_err_peak_region = compute_mean_abs_error_region(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical], rogue_region)
    
    print(f"RMSE (domain) at t={selected_time_s:.2f}: {rmse_at_target:.6f}")
    if rogue_region is None:
        print(f"No rogue region detected")
    else:
        print(f"Mean absolute error in rogue region: {mean_abs_err_peak_region:.6f}")
    
                                
    if save_data:
                          
        time_series_data = {
            'time': time_vec,
            'mse': mse,
            'residual_mean': residual_mean,
            'spread': spread
        }
        time_series_df = pd.DataFrame(time_series_data)
        time_series_path = os.path.join(exp_dir, f'time_series_{title_suffix}.csv')
        time_series_df.to_csv(time_series_path, index=False)
        
                                          
        prediction_data = {
            'x': x,
            'truth': selected_truth,
            'prediction': selected_estimate,
            'std': selected_std,
            'lower_2sigma': selected_estimate - 2*selected_std,
            'upper_2sigma': selected_estimate + 2*selected_std
        }
        prediction_df = pd.DataFrame(prediction_data)
        prediction_path = os.path.join(exp_dir, f'prediction_data_t{selected_time_s:.2f}_{title_suffix}.csv')
        prediction_df.to_csv(prediction_path, index=False)
        
                         
        metrics_data = {
            'target_time': [target_time],
            'analysis_time': [analysis_time],
            'lead_time': [lead_time],
            'selected_time': [selected_time_s],
            'N_ensemble': [N_ensemble_actual],
            'n_prediction_ensemble': [n_pred_actual],
            'Q_d': [Q_d],
            'rmse_domain': [rmse_at_target],
            'peak_amplitude_error': [peak_amp_error],
            'mean_abs_error_peak_region': [mean_abs_err_peak_region],
            'final_mse': [mse[-1] if len(mse) > 0 else np.nan],
            'final_spread': [spread[-1] if len(spread) > 0 else np.nan]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(exp_dir, f'metrics_{title_suffix}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"Results saved to {exp_dir}")
    
    return results, exp_dir


def compare_alpha_values_unet(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                             alpha_values, selected_members=None, n_prediction_ensemble=None, 
                             save_data=True, time_chosen=None):
    base_dir = r"C:\\Users\\DELL\\Desktop\\Thesis code\\KdV code\\Numerical sol\\Final code\\Final_Final_code\\Data_assimilation\\ENKF_Unet_newresults"
    os.makedirs(base_dir, exist_ok=True)
    
    if n_prediction_ensemble is None:
        n_prediction_ensemble = N_ensemble
    
    compare_name = f"compare_alpha_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time}_npred{n_prediction_ensemble}"
    compare_dir = os.path.join(base_dir, compare_name)
    os.makedirs(compare_dir, exist_ok=True)
    
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    
    results_dict = {}
    
    print(f"Running alpha comparison for {len(alpha_values)} alpha values...")
    
    for i, alpha in enumerate(alpha_values):
        print(f"\\nRunning experiment {i+1}/{len(alpha_values)}: alpha={alpha}")
        
        results = run_enkf_enhanced_unet_experiment(
            target_time, analysis_time, lead_time, N_ensemble, Q_d,
            selected_members, alpha, n_prediction_ensemble, save_data
        )
        
        results_dict[alpha] = results
    
    print("Generating comparison plots...")
    
                            
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))
    
    for alpha, color in zip(alpha_values, colors):
        results = results_dict[alpha]
        time_vec = np.arange(len(results['mse_total'])) * results['dt_numerical']
        
        sample_step = 1000
        mask_upto = time_vec <= (selected_time_s + 1e-12)
        t_plot = time_vec[mask_upto][::sample_step]
        mse_plot = results['mse_total'][mask_upto][::sample_step]
        
        plt.semilogy(t_plot, mse_plot, color=color, linewidth=2.5, label=f'α={alpha}')
    
    plt.axvline(x=analysis_time, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Analysis time (t={analysis_time:.1f})')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Target time (t={selected_time_s:.1f})')
    
    plt.xlabel('Time t')
    plt.ylabel('MSE')
    plt.title(f'MSE Evolution Comparison - Alpha Values (N={N_ensemble}, npred={n_prediction_ensemble})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, selected_time_s)
                          
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'mse_evolution_alpha_comparison_t{selected_time_s:.2f}.png'), dpi=300)
               
    
                                                         
    if save_data:
        for alpha in alpha_values:
            results = results_dict[alpha]
            time_vec = np.arange(len(results['mse_total'])) * results['dt_numerical']
            
            alpha_data = {
                'time': time_vec,
                'mse': results['mse_total'],
                'residual_mean': results['residual_mean_total'],
                'spread': results['spread_total']
            }
            alpha_df = pd.DataFrame(alpha_data)
            alpha_path = os.path.join(compare_dir, f'time_series_alpha{alpha}.csv')
            alpha_df.to_csv(alpha_path, index=False)
    
    print(f"Alpha comparison results saved to {compare_dir}")
    return results_dict, compare_dir

def compare_n_prediction_values_unet(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                                    n_prediction_values, selected_members=None, alpha=1.0, 
                                    save_data=True, time_chosen=None):
    base_dir = r"C:\\Users\\DELL\\Desktop\\Thesis code\\KdV code\\Numerical sol\\Final code\\Final_Final_code\\Data_assimilation\\ENKF_Unet_newresults"
    os.makedirs(base_dir, exist_ok=True)
    
    compare_name = f"Npred_comparison_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}_lead{lead_time:.1f}_alpha{alpha}"
    compare_dir = os.path.join(base_dir, compare_name)
    os.makedirs(compare_dir, exist_ok=True)
    
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    
    results_dict = {}
    
    print(f"Running n_prediction comparison for {len(n_prediction_values)} values...")
    
    for i, n_pred in enumerate(n_prediction_values):
        print(f"\nRunning experiment {i+1}/{len(n_prediction_values)}: n_prediction={n_pred}")
        
        results = run_enkf_enhanced_unet_experiment(
            target_time, analysis_time, lead_time, N_ensemble, Q_d,
            selected_members, alpha, n_pred, save_data
        )
        
        results_dict[n_pred] = results
    
    print("\nGenerating comparison plots...")
    
                            
    plt.figure(figsize=(12, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(n_prediction_values)))
    
    for n_pred, color in zip(n_prediction_values, colors):
        results = results_dict[n_pred]
        time_vec = np.arange(len(results['mse_total'])) * results['dt_numerical']
        
        sample_step = 1000
        mask_upto = time_vec <= (selected_time_s + 1e-12)
        t_plot = time_vec[mask_upto][::sample_step]
        mse_plot = results['mse_total'][mask_upto][::sample_step]
        
        plt.semilogy(t_plot, mse_plot, color=color, linewidth=2.5, label=f'n_pred={n_pred}')
    
    plt.axvline(x=analysis_time, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Analysis time (t={analysis_time:.1f})')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Target time (t={selected_time_s:.1f})')
    
    plt.xlabel('Time t', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title(f'MSE Evolution Comparison: Different N_pred Values(α={alpha}, Analysis until t={analysis_time:.1f})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, selected_time_s)
                          
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'mse_evolution_npred_comparison_t{selected_time_s:.2f}.png'), dpi=300)
                            

                                                                      
    for i, n_pred in enumerate(n_prediction_values):
        results = results_dict[n_pred]
        x = results['x']
        dt_numerical = results['dt_numerical']
        selected_idx = int(round(selected_time_s / dt_numerical))
        selected_idx = max(0, min(selected_idx, len(results['u_truth']) - 1))
        selected_truth = results['u_truth'][selected_idx]
        if selected_idx <= results['analysis_idx']:
            ensemble_snapshot = results['ensemble_final_da']
            phase_info_single = 'Data Assimilation'
        else:
            ensemble_snapshot = results['ensemble_final_pred']
            phase_info_single = 'Pure Prediction'
        plt.figure(figsize=(14, 8))
        plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=5)
        num_members = ensemble_snapshot.shape[1]
        member_colors = plt.cm.viridis(np.linspace(0, 1, num_members))
        for j in range(num_members):
            plt.plot(x, ensemble_snapshot[:, j], color=member_colors[j], linewidth=1.2, alpha=0.6)
        plt.xlabel('Spatial Position x', fontsize=14)
        plt.ylabel('Wave Amplitude u', fontsize=14)
        plt.title(f'Ensemble Members vs Truth (N_pred={n_pred}) at t = {selected_time_s:.2f} ({phase_info_single})' + f' (α={alpha})', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(compare_dir, f'ensemble_members_vs_truth_npred_{n_pred}_t_{selected_time_s:.2f}.png'), dpi=300, bbox_inches='tight')
                                
    
                                       
    if save_data:
        for n_pred in n_prediction_values:
            results = results_dict[n_pred]
            time_vec = np.arange(len(results['mse_total'])) * results['dt_numerical']
            
            npred_data = {
                'time': time_vec,
                'mse': results['mse_total'],
                'residual_mean': results['residual_mean_total'],
                'spread': results['spread_total']
            }
            npred_df = pd.DataFrame(npred_data)
            npred_path = os.path.join(compare_dir, f'time_series_npred{n_pred}.csv')
            npred_df.to_csv(npred_path, index=False)
    
    print(f"N_prediction comparison results saved to {compare_dir}")
    return results_dict, compare_dir

def compare_alpha_npred_combinations_unet(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                                         alpha_values, n_prediction_values, selected_members=None, 
                                         save_data=True, time_chosen=None):
    base_dir = r"C:\\Users\\DELL\\Desktop\\Thesis code\\KdV code\\Numerical sol\\Final code\\Final_Final_code\\Data_assimilation\\ENKF_Unet_newresults"
    os.makedirs(base_dir, exist_ok=True)
    
    compare_name = f"Alpha_Npred_comparison_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}_lead{lead_time:.1f}"
    compare_dir = os.path.join(base_dir, compare_name)
    os.makedirs(compare_dir, exist_ok=True)
    
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    
    results_dict = {}
    rmse_grid = np.zeros((len(alpha_values), len(n_prediction_values)))
    
    total_combinations = len(alpha_values) * len(n_prediction_values)
    print(f"Running {total_combinations} combinations...")
    
    combination_idx = 0
    for i, alpha in enumerate(alpha_values):
        for j, n_pred in enumerate(n_prediction_values):
            combination_idx += 1
            print(f"\nRunning combination {combination_idx}/{total_combinations}: α={alpha}, n_pred={n_pred}")

            np.random.seed(40)
            torch.manual_seed(40)
            random.seed(40)
            
            results = run_enkf_enhanced_unet_experiment(
                target_time, analysis_time, lead_time, N_ensemble, Q_d,
                selected_members, alpha, n_pred, save_data
            )
            
            key = (alpha, n_pred)
            results_dict[key] = results
            
                                        
            selected_idx = int(round(selected_time_s / results['dt_numerical']))
            selected_idx = max(0, min(selected_idx, len(results['u_truth']) - 1))
            rmse = compute_rmse_over_domain(results['u_truth'][selected_idx], results['x_mean_total'][selected_idx])
            rmse_grid[i, j] = rmse
            
                                                              
            create_enhanced_unet_figures(
                target_time, analysis_time, lead_time, N_ensemble, Q_d,
                selected_members, alpha, n_pred, save_data, time_chosen
            )
                                                                                           
            try:
                selected_idx = int(round(selected_time_s / results['dt_numerical']))
                selected_idx = max(0, min(selected_idx, len(results['u_truth']) - 1))
                x = results['x']
                truth = results['u_truth'][selected_idx]
                est = results['x_mean_total'][selected_idx]
                plt.figure(figsize=(12, 6))
                plt.plot(x, truth, 'k-', lw=2.5, label='Truth')
                plt.plot(x, est, 'r--', lw=2.5, label='EnKF-UNet')
                plt.xlabel('Spatial Position x', fontsize=14)
                plt.ylabel('Wave Amplitude u', fontsize=14)
                plt.title(f'Prediction α{alpha}_N{n_pred}_t_{selected_time_s:.2f}', fontsize=16, fontweight='bold')
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(compare_dir, f'prediction_α{alpha}_N{n_pred}_t_{selected_time_s:.2f}.png'), dpi=300)
                                        
            except Exception as _:
                pass
    
    print("\nGenerating combined comparison plots...")
    
                  
    plt.figure(figsize=(12, 8))
    im = plt.imshow(rmse_grid, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='RMSE')
    
    plt.xticks(range(len(n_prediction_values)), n_prediction_values)
    plt.yticks(range(len(alpha_values)), alpha_values)
    plt.xlabel('N_prediction_ensemble', fontsize=14)
    plt.ylabel('Alpha (α)', fontsize=14)
    plt.title(f'RMSE Heatmap at t = {selected_time_s:.2f}', fontsize=16, fontweight='bold')
    
                          
    for i in range(len(alpha_values)):
        for j in range(len(n_prediction_values)):
            plt.text(j, i, f'{rmse_grid[i, j]:.4f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'rmse_heatmap_combinations_t_{selected_time_s:.2f}.png'), 
                dpi=300, bbox_inches='tight')
                            
    
    print(f"Combination comparison results saved to {compare_dir}")
    return results_dict, compare_dir

if __name__ == "__main__":        
                                   
                                                             
                                                                    
                                                                    
       
    
                  
                                                  
                                                            
                                                                         
                                                             
                                                            
                                                     
       
    
                          
                                              
                                                                               
                                                                         
                                                                          
                                                                                       
       
    
    alpha_values = [1.0]
    n_prediction_values = [1,10,50,150]
    
    results_dict_combo, compare_dir_combo = compare_alpha_npred_combinations_unet(
        target_time=15.0,          
        analysis_time=6.0,        
        lead_time=7.8,            
        N_ensemble=150,            
        Q_d=10,                   
        alpha_values=alpha_values,
        n_prediction_values=n_prediction_values,
        selected_members=None,     
        save_data=True,           
        time_chosen=7.8            
    )

