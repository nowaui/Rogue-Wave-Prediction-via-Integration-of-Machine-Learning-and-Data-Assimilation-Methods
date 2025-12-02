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

def run_enkf_unet_experiment(lead_time, N_ensemble, Q_d, analysis_time=None, obs_ratio_override=None, obs_noise_std_override=None, Q_std_override=None):
                       
    L = 50.0
    dx = 0.1
    M = int(L / dx)
    dt_numerical = 1e-4
    dt_frame = 0.1
    
    T_numerical = lead_time + 0.1
    N_t_numerical = int(T_numerical / dt_numerical)
    N_t_frame = int(lead_time / dt_frame)
    
    x = np.linspace(-L/2, L/2, M, endpoint=False)
    lead_idx_numerical = int(lead_time / dt_numerical)
    
                     
    device = torch.device('cpu')
    m = 4
    model_path = get_model_path(exp=30, batch_size=64, m=m)
    
    print(f"Running EnKF-UNet: lead_time={lead_time:.1f}s, N={N_ensemble}, Q_d={Q_d}")
    if analysis_time is not None:
        print(f"Analysis cutoff time: {analysis_time:.2f}s (pure prediction after this)")
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
        else:
            raise ValueError("Truth data dimensions don't match expected shape.")
    else:
        raise FileNotFoundError(
            f"Truth solution file {store_filename} not found in {store_dir}. "
            "Please generate it separately before running this experiment."
        )
    
    u_min, u_max = np.min(u_truth), np.max(u_truth)
    
                        
    obs_ratio = 0.1 if obs_ratio_override is None else float(obs_ratio_override)
    obs_interval = 3 

    ratio = int(dt_frame / dt_numerical)
    sqrt_ratio = np.sqrt(ratio)  
    obs_noise_std = 0.1 if obs_noise_std_override is None else float(obs_noise_std_override)
    
    n_obs = int(obs_ratio * M)
    obs_indices = np.sort(np.random.choice(M, n_obs, replace=False))
    
    H = np.zeros((n_obs, M))
    for i, idx in enumerate(obs_indices):
        H[i, idx] = 1.0
    R = (obs_noise_std**2) * np.eye(n_obs)
    
                                                                 
    observations = {}
    cutoff_frame_idx = N_t_frame + 1                      
    if analysis_time is not None:
        cutoff_frame_idx = int(analysis_time / dt_frame)
        print(f"Observations will stop at frame index {cutoff_frame_idx} (t={cutoff_frame_idx*dt_frame:.3f}s)")
    
    for k in range(0, N_t_frame + 1, obs_interval):
        if k <= cutoff_frame_idx:                                            
            numerical_idx = int(k * dt_frame / dt_numerical)
            if numerical_idx < len(u_truth):
                y_true = u_truth[numerical_idx, obs_indices]
                noise = obs_noise_std * np.random.normal(0, 1, n_obs)
                observations[k] = y_true + noise
    
    print(f"Generated {len(observations)} observation points")
    
                     
    unet_system = UNetSystemK1(unet_model, device, m=m, u_min=u_min, u_max=u_max, dx=dx, dt_frame=dt_frame, nu=0.01)
    Q_std_effective = sqrt_ratio * 5e-4 if Q_std_override is None else float(Q_std_override)
    enkf_unet = EnKF_UNet(unet_system, N_ensemble, Q_std= Q_std_effective, R_std=obs_noise_std, Q_d=Q_d)
    
                                                          
    x0_zero = np.zeros_like(u_truth[0])
    initial_conditions = [("zeros", x0_zero)]
    
    results = {}
    
    for ic_name, x0_est in initial_conditions:
        print(f"\nRunning EnKF with initial condition: {ic_name}")
        
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
                if z_k is not None and frame_idx <= cutoff_frame_idx:
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
        print(f"Completed {ic_name} in {running_time:.2f}s")
        print(f"Filled {np.sum(filled_indices)} out of {len(filled_indices)} time points")
        
                                                  
        results[ic_name] = {
            'x_mean': x_mean_history,
            'std': std_history,
            'mse': mse_history,
            'residual_mean': residual_mean_history,
            'spread': spread_history,
            'filled_indices': filled_indices,
            'running_time': running_time
        }
    
                                      
                         
                                            
                                                                      
                                                          
                                                                                                      
        
                       
                                                  
                                                                                                   

        
                               
                                      
                                        
                                       
                                                   
                                     
                                                                                           
                
                                  
                                                     
                                                                                         
                                                                          
                
                                                                          
                                                                                                             
                                                      
                                                
                                                                                    
                                                                                         
                                                            
    
                                                
    return {
        'u_truth': u_truth,
        'results': results,                                                
        'x': x,
        'lead_idx_numerical': lead_idx_numerical,
        'dt_frame': dt_frame,
        'dt_numerical': dt_numerical,
        'cutoff_frame_idx': cutoff_frame_idx,
        'analysis_time': analysis_time
    }

def create_all_figures_unet(lead_time, N_ensemble, Q_d, time_chosen=None, analysis_time=None):
    save_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_Unet_results\para_compare"
    os.makedirs(save_dir, exist_ok=True)
    

    base = run_enkf_unet_experiment(lead_time, N_ensemble, Q_d, analysis_time)

    var = run_enkf_unet_experiment(lead_time, N_ensemble, Q_d, analysis_time,
                                   obs_ratio_override=0.2, obs_noise_std_override=0.1, Q_std_override=1e-3)
    
                       
    u_truth = base['u_truth']
    x = base['x']
    dt_numerical = base['dt_numerical']
    analysis_time_used = base['analysis_time']
    
    base_truth = base['results']['zeros']
    var_truth = var['results']['zeros']
    
    time_vec = np.arange(len(base_truth['mse'])) * dt_numerical

                                                                              
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
                                                 
    selected_idx_numerical = int(round(selected_time_s / dt_numerical))
    selected_idx_numerical = max(0, min(selected_idx_numerical, len(u_truth) - 1))
                                        
    selected_truth = u_truth[selected_idx_numerical]
    base_estimate_truth = base_truth['x_mean'][selected_idx_numerical]
    var_estimate_truth = var_truth['x_mean'][selected_idx_numerical]
    base_std_truth = base_truth['std'][selected_idx_numerical]
    var_std_truth = var_truth['std'][selected_idx_numerical]
    
                                                                    
    sample_step = 1000
    mask_upto = time_vec <= (selected_time_s + 1e-12)
    t_plot = time_vec[mask_upto][::sample_step]
    base_mse = base_truth['mse'][mask_upto][::sample_step]
    var_mse = var_truth['mse'][mask_upto][::sample_step]
    base_residual = base_truth['residual_mean'][mask_upto][::sample_step]
    var_residual = var_truth['residual_mean'][mask_upto][::sample_step]
    base_spread = base_truth['spread'][mask_upto][::sample_step]
    var_spread = var_truth['spread'][mask_upto][::sample_step]
    
                                               
    analysis_str = f"_analysis{analysis_time_used:.1f}s" if analysis_time_used is not None else "_no_analysis"
    
                                                       
    plt.figure(figsize=(12, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=3, label='Truth')
    plt.plot(x, base_estimate_truth, 'g:', linewidth=2.5, label='EnKF-UNet (base)')
    plt.plot(x, var_estimate_truth, 'm--', linewidth=2.5, label='EnKF-UNet (variant)')
    
                                               
    if analysis_time_used is not None:
        plt.plot([], [], ' ', label=f'Analysis cutoff = {analysis_time_used:.1f}s')
    
    plt.xlabel('Spatial Position x')
    plt.ylabel('Wave Amplitude u')
    plt.title(f'Prediction (Zero IC) Param Comparison at t={selected_time_s:.1f}s', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_param_comp_t{selected_time_s:.2f}s{analysis_str}_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
                
    plt.close()
    
                                      
    plt.figure(figsize=(12, 6))
    plt.semilogy(t_plot, base_mse, 'g-', linewidth=2.5, label='MSE (base)')
    plt.semilogy(t_plot, var_mse, 'm--', linewidth=2.5, label='MSE (variant)')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, label=f'Lead time t={selected_time_s:.1f}s')
    
                                   
    if analysis_time_used is not None:
        plt.axvline(x=analysis_time_used, color='orange', linestyle=':', linewidth=3, 
                    label=f'Analysis cutoff (t={analysis_time_used:.2f}s)')
                                                     
        plt.axvspan(analysis_time_used, selected_time_s, alpha=0.2, color='orange')
    
    plt.xlabel('Time t (s)')
    plt.xlim(0.0, selected_time_s)
    plt.ylim(0.001, 0.1)
    plt.ylabel('MSE')
    plt.title('MSE Evolution (Zero IC) - Param Comparison', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    sampled_mse_full_base = base_truth['mse'][mask_upto]
    sampled_mse_full_var = var_truth['mse'][mask_upto]
    
                           
    valid_mse_mask_base = np.isfinite(sampled_mse_full_base)
    valid_mse_mask_var = np.isfinite(sampled_mse_full_var)
    
    ax = plt.gca()
    text_lines = []
    if np.any(valid_mse_mask_base):
        rmse_final_base = float(np.sqrt(sampled_mse_full_base[valid_mse_mask_base][-1]))
        text_lines.append(f'Final RMSE (base) = {rmse_final_base:.6f}')
    if np.any(valid_mse_mask_var):
        rmse_final_var = float(np.sqrt(sampled_mse_full_var[valid_mse_mask_var][-1]))
        text_lines.append(f'Final RMSE (variant) = {rmse_final_var:.6f}')
    
    if text_lines:
        ax.text(0.98, 0.92, '\n'.join(text_lines), transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mse_evolution_param_comp_t{selected_time_s:.2f}s{analysis_str}_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
                
    plt.close()
    
                                                     
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, base_residual, 'g-', linewidth=2.5, label='Mean Residual (base)')
    plt.plot(t_plot, var_residual, 'm--', linewidth=2.5, label='Mean Residual (variant)')
    plt.axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=selected_time_s, color='blue', linestyle='--', linewidth=2, label=f'Lead time t={selected_time_s:.1f}s')
    
                                   
    if analysis_time_used is not None:
        plt.axvline(x=analysis_time_used, color='orange', linestyle=':', linewidth=3, 
                    label=f'Analysis cutoff (t={analysis_time_used:.2f}s)')
                                                     
        plt.axvspan(analysis_time_used, selected_time_s, alpha=0.2, color='orange')
    
    plt.xlabel('Time t (s)')
    plt.ylabel('Mean Residual')
    plt.title('Mean Residual Evolution (Zero IC) - Param Comparison', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'residual_evolution_param_comp_t{selected_time_s:.2f}s{analysis_str}_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
                
    plt.close()
    
                                                            
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, base_spread, 'g-', linewidth=2.5, label='Spread (base)')
    plt.plot(t_plot, var_spread, 'm--', linewidth=2.5, label='Spread (variant)')
    plt.axvline(x=selected_time_s, color='blue', linestyle='--', linewidth=2, label=f'Lead time t={selected_time_s:.1f}s')
    
                                   
    if analysis_time_used is not None:
        plt.axvline(x=analysis_time_used, color='orange', linestyle=':', linewidth=3, 
                    label=f'Analysis cutoff (t={analysis_time_used:.2f}s)')
                                                     
        plt.axvspan(analysis_time_used, selected_time_s, alpha=0.2, color='orange')
    
    plt.xlabel('Time t (s)')
    plt.ylabel('Spread')
    plt.title('Spread Evolution (Zero IC) - Param Comparison', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
                             
    sampled_spread_full_base = base_truth['spread'][mask_upto]
    sampled_spread_full_var = var_truth['spread'][mask_upto]
    
    valid_spread_mask_base = np.isfinite(sampled_spread_full_base)
    valid_spread_mask_var = np.isfinite(sampled_spread_full_var)
    
    ax = plt.gca()
    text_lines = []
    if np.any(valid_spread_mask_base):
        spread_final_base = float(sampled_spread_full_base[valid_spread_mask_base][-1])
        text_lines.append(f'Final Spread (base) = {spread_final_base:.6f}')
    if np.any(valid_spread_mask_var):
        spread_final_var = float(sampled_spread_full_var[valid_spread_mask_var][-1])
        text_lines.append(f'Final Spread (variant) = {spread_final_var:.6f}')
    
    if text_lines:
        ax.text(0.98, 0.92, '\n'.join(text_lines), transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spread_evolution_param_comp_t{selected_time_s:.2f}s{analysis_str}_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
                
    plt.close()
    
                                              
    plt.figure(figsize=(16, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=3, label='Truth')
    plt.plot(x, base_estimate_truth, 'g:', linewidth=2.5, label='EnKF-UNet (base)')
    lower_b = base_estimate_truth - 2*base_std_truth
    upper_b = base_estimate_truth + 2*base_std_truth
    plt.fill_between(x, lower_b, upper_b, alpha=0.2, color='green', label='2σ (base)')
    plt.plot(x, var_estimate_truth, 'm--', linewidth=2.5, label='EnKF-UNet (variant)')
    lower_v = var_estimate_truth - 2*var_std_truth
    upper_v = var_estimate_truth + 2*var_std_truth
    plt.fill_between(x, lower_v, upper_v, alpha=0.2, color='magenta', label='2σ (variant)')
    
    plt.xlabel('Spatial Position x')
    plt.ylabel('Wave Amplitude u')
    plt.title(f'Error Bands (Zero IC) Param Comparison at t={selected_time_s:.1f}s', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'error_bands_param_comp_t{selected_time_s:.1f}s{analysis_str}_N{N_ensemble}_Qd{Q_d}.png'), dpi=300)
                
    plt.close()

                                                    
    rmse_at_lead_truth = compute_rmse_over_domain(u_truth[selected_idx_numerical], base_estimate_truth)

    rogue_threshold = 1.0
    rogue_region = u_truth[selected_idx_numerical] > rogue_threshold
    if not np.any(rogue_region):
        rogue_region = None
    
                                                 
    peak_amp_error_truth = compute_peak_amplitude_error(u_truth[selected_idx_numerical], base_estimate_truth, rogue_region)
    mean_abs_err_peak_region_truth = compute_mean_abs_error_region(u_truth[selected_idx_numerical], base_estimate_truth, rogue_region)

    comp_time_sec_truth = float(base_truth['running_time'])

    print(f"RMSE (domain) at t={selected_time_s:.2f}s:")
    print(f"  Zero IC: {rmse_at_lead_truth:.6f}")
    if rogue_region is None:
        print(f"No rogue region detected")
    else:
        print(f"Mean absolute error in rogue region:")
        print(f"  Zero IC: {mean_abs_err_peak_region_truth:.6f}")
    print(f"Computational time (s):")
    print(f"  Zero IC: {comp_time_sec_truth:.2f}")

                                                     
    if analysis_time_used is not None:
        csv_filename = f'metrics_summary_Unet_analysis_{analysis_time_used:.1f}s.csv'
    else:
        csv_filename = 'metrics_summary_Unet_no_analysis.csv'
    metrics_csv = os.path.join(save_dir, csv_filename)
    
                              
    metrics_rows = []
    
                 
    metrics_rows.append({
        'initial_condition': 'zeros',
        'lead_time_s': float(selected_time_s),
        'run_lead_time_s': float(lead_time),
        'N_ensemble': int(N_ensemble),
        'Q_d': float(Q_d),
        'rmse_domain': rmse_at_lead_truth,
        'peak_amplitude_error': peak_amp_error_truth,
        'computational_time_s': comp_time_sec_truth,
        'mean_abs_error_peak_region': mean_abs_err_peak_region_truth
    })
    
    if os.path.exists(metrics_csv):
        try:
            df_metrics = pd.read_csv(metrics_csv)
        except Exception:
            df_metrics = pd.DataFrame()
        df_metrics = pd.concat([df_metrics, pd.DataFrame(metrics_rows)], ignore_index=True)
    else:
        df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(metrics_csv, index=False)

    print(f"Results saved to {save_dir}")
    print(f"Final MSE:")
    print(f"  Zero IC: {base_truth['mse'][-1]:.6f}")
    print(f"Final Mean Residual:")
    print(f"  Zero IC: {base_truth['residual_mean'][-1]:.6f}")
    print(f"Final Ensemble Spread:")
    print(f"  Zero IC: {base_truth['spread'][-1]:.6f}")


def plot_metrics_vs_analysis():
    data_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_Unet_results\Analysis"
    save_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_Unet_results\Analysis_Results"
    
    import glob
    import re
    
    pattern = os.path.join(data_dir, 'metrics_summary_Unet_analysis_*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No analysis CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    for csv_file in csv_files:
        try:
            filename = os.path.basename(csv_file)
            match = re.search(r'metrics_summary_Unet_analysis_(\d+\.\d+)s\.csv', filename)
            if not match:
                continue
            
            analysis_value = float(match.group(1))
            df = pd.read_csv(csv_file)
            
            if len(df) > 0:
                                                 
                for _, row in df.iterrows():
                    ic_type = row.get('initial_condition', 'unknown')
                    data_point = {
                        'analysis_time': analysis_value,
                        'initial_condition': ic_type,
                        'rmse_domain': row.get('rmse_domain', np.nan),
                        'peak_amplitude_error': row.get('peak_amplitude_error', np.nan),
                        'mean_abs_error_peak_region': row.get('mean_abs_error_peak_region', np.nan),
                        'computational_time_s': row.get('computational_time_s', np.nan)
                    }
                    all_data.append(data_point)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not all_data:
        print("No valid data found!")
        return
    
    df = pd.DataFrame(all_data)
    df = df.sort_values('analysis_time').reset_index(drop=True)
    
    unique_analysis_times = df['analysis_time'].nunique() if not df.empty else 0
    total_data_points = len(df)
    print(f"Collected {total_data_points} data points for {unique_analysis_times} analysis times (2 initial conditions each)")
    
    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ('rmse_domain', 'RMSE (Domain)'),
        ('peak_amplitude_error', 'Peak Amplitude Error'),
        ('mean_abs_error_peak_region', 'Peak Region MAE'),
        ('computational_time_s', 'Computational Time (s)')
    ]
    
    print("Creating metric plots...")
    
    for metric_key, ylabel in metrics:
        valid_data = df.dropna(subset=[metric_key])
        if valid_data.empty:
            print(f"  Skipping {metric_key}: no valid data")
            continue
        
        plt.figure(figsize=(10, 6))
        
                                                 
        for ic_type in ['zeros']:
            ic_data = valid_data[valid_data['initial_condition'] == ic_type]
            if not ic_data.empty:
                label = 'Zero IC' if ic_type == 'zeros' else 'Truth IC'
                marker = 'o' if ic_type == 'zeros' else 's'
                color = 'red' if ic_type == 'zeros' else 'green'
                plt.plot(ic_data['analysis_time'], ic_data[metric_key], 
                        marker + '-', linewidth=2, markersize=6, label=label, color=color)
        
        plt.xlabel('Analysis Time (s)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{ylabel} vs Analysis Time (Comparison)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
 
        filename = f'{metric_key}_vs_analysis_comparison.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.show()
        plt.close()
    
    print(f"All plots saved to: {save_dir}")

def plot_rmse_vs_obs_ratio(lead_time, N_ensemble, Q_d, obs_ratios, time_chosen=None, analysis_time=None):
    """
    Sweep observation ratios and plot RMSE at the requested evaluation time.
    """
    if not obs_ratios:
        print("No obs_ratio values provided for sweep.")
        return

    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    ic_labels = {
        'zeros': 'Zero IC'
    }

    rmse_by_ic = {}

    print(f"\nStarting obs_ratio sweep for RMSE at t={selected_time_s:.2f}s ...")
    for obs_ratio in obs_ratios:
        print(f"\n>> obs_ratio = {obs_ratio}")
        exp_results = run_enkf_unet_experiment(
            lead_time,
            N_ensemble,
            Q_d,
            analysis_time=analysis_time,
            obs_ratio_override=obs_ratio
        )

        dt_numerical = exp_results['dt_numerical']
        idx = int(round(selected_time_s / dt_numerical))
        idx = max(0, min(idx, len(exp_results['u_truth']) - 1))

        for ic_name, stats in exp_results['results'].items():
            mse_series = stats['mse']
            if idx >= len(mse_series):
                rmse_val = np.nan
            else:
                rmse_val = float(np.sqrt(mse_series[idx]))
            rmse_by_ic.setdefault(ic_name, []).append(rmse_val)
            print(f"   {ic_labels.get(ic_name, ic_name)} RMSE = {rmse_val:.6f}")

    save_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_Unet_results\obs_ratio_sweep"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    marker_styles = ['o', 's', '^', 'd']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, (ic_name, values) in enumerate(rmse_by_ic.items()):
        plt.plot(
            obs_ratios,
            values,
            marker_styles[i % len(marker_styles)] + '-',
            label=ic_labels.get(ic_name, ic_name),
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6
        )

    plt.xlabel('Observation Ratio', fontsize=12)
    plt.ylabel('RMSE (domain)', fontsize=12)
    title_extra = f"(analysis cutoff={analysis_time:.1f}s)" if analysis_time is not None else "(no cutoff)"
    plt.title(f'RMSE vs Observation Ratio at t={selected_time_s:.2f}s {title_extra}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f'rmse_vs_obs_ratio_t{selected_time_s:.2f}_N{N_ensemble}_Qd{Q_d}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RMSE vs Obs Ratio plot saved to: {filepath}")

if __name__ == "__main__":
    lead_time_main = 6.0
    analysis_time = 6.0
    N_ensemble = 150
    Q_d = 10

    create_all_figures_unet(15.0, N_ensemble, Q_d, time_chosen=lead_time_main, analysis_time=analysis_time)

    obs_ratio_values = [0.05, 0.1, 0.15, 0.2, 0.3]
    plot_rmse_vs_obs_ratio(
        lead_time=lead_time_main,
        N_ensemble=N_ensemble,
        Q_d=Q_d,
        obs_ratios=obs_ratio_values,
        time_chosen=lead_time_main,
        analysis_time=analysis_time
    )

