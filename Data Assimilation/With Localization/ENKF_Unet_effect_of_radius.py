import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import pandas as pd
import random
from scipy.spatial.distance import cdist

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

                                   
def gaspari_cohn(r, c):
    r = np.abs(r)
    r_norm = r / c
    
                             
    rho = np.zeros_like(r_norm)
    
                           
    mask1 = (r_norm >= 0) & (r_norm <= 1)
    r1 = r_norm[mask1]
    rho[mask1] = 1 - (5/3)*r1**2 + (5/8)*r1**3 + (1/2)*r1**4 - (1/4)*r1**5
    
                          
    mask2 = (r_norm > 1) & (r_norm <= 2)
    r2 = r_norm[mask2]
    rho[mask2] = 4 - 5*r2 + (5/3)*r2**2 + (5/8)*r2**3 - (1/2)*r2**4 + (1/12)*r2**5 - 2/(3*r2)
    
                     
    mask3 = r_norm > 2
    rho[mask3] = 0
    
    return rho

def periodic_distance(i, j, M, dx):
                               
    direct_dist = np.abs(i - j) * dx
    
                                
    wrapped_dist = (M - np.abs(i - j)) * dx
    
                             
    return np.minimum(direct_dist, wrapped_dist)

def create_localization_matrix(M, dx, localization_radius):
    rho = np.zeros((M, M))
    
    for i in range(M):
        for j in range(M):
                                         
            dist = periodic_distance(i, j, M, dx)
                                             
            rho[i, j] = gaspari_cohn(dist, localization_radius)
    
    return rho

             
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
    def __init__(self, unet_system, N, Q_std, R_std, Q_d, localization_radius=None, dx=0.1):
        self.unet_system = unet_system
        self.M_dim = unet_system.M
        self.N = N
        self.Q_std = Q_std
        self.R_std = R_std
        self.dx = dx
        
        self.Q = self.create_gaussian_Q(Q_d)
        self.L_Q = np.linalg.cholesky(self.Q)
        self.G = np.eye(self.M_dim)
                                                                       
        self.debug_save_dir = None
        self._p_loc_dump_counter = 0
        
                                 
        self.localization_radius = localization_radius
        if localization_radius is not None:
            self.localization_matrix = create_localization_matrix(self.M_dim, dx, localization_radius)
            print(f"Covariance localization enabled with radius: {localization_radius}")
            print(f"Localization matrix shape: {self.localization_matrix.shape}")
        else:
            self.localization_matrix = None
            print("No covariance localization applied")
    
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
                                                      
        denom = np.sqrt(max(self.N - 1, 1))
        L_f = xi_anomaly / denom
        return x_f, L_f
    
    def measurement_update(self, xi_f, z_k, H, R, print_covariance=False):
        x_f, L_f = self.compute_mean_anomaly(xi_f)
        
        P_f = L_f @ L_f.T
        
        
                                                                      
        if self.localization_matrix is not None:
            P_f_loc = P_f * self.localization_matrix
        else:
            P_f_loc = P_f

                                                         
                               
        S = H @ P_f_loc @ H.T + R
        K_k = P_f_loc @ H.T @ np.linalg.inv(S)
        
        v_k = self.R_std * np.random.randn(len(z_k), self.N)
        H_xi_f = H @ xi_f
        innovations = z_k[:, np.newaxis] - H_xi_f - v_k
        xi_a = xi_f + K_k @ innovations
        
        if print_covariance:
                                                                                                
            print(f"P_loc shape: {P_f_loc.shape}")
            try:
                mid = self.M_dim // 2
                names = [
                    ("P_loc[0,0]", P_f_loc[0,0]),
                    ("P_loc[0,1]", P_f_loc[0,1]),
                    ("P_loc[0,-1]", P_f_loc[0,-1]),
                    ("P_loc[1,0]", P_f_loc[1,0]),
                    ("P_loc[-1,0]", P_f_loc[-1,0]),
                    (f"P_loc[{mid},{(mid+1)%self.M_dim}]", P_f_loc[mid,(mid+1)%self.M_dim]),
                    (f"P_loc[{mid},{(mid-1)%self.M_dim}]", P_f_loc[mid,(mid-1)%self.M_dim])
                ]
                for n,v in names:
                    print(f"{n} = {v:.6e}")
            except Exception:
                pass
                                                
            try:
                if self.debug_save_dir is not None:
                    os.makedirs(self.debug_save_dir, exist_ok=True)
                    csv_name = "P_loc.csv" if self._p_loc_dump_counter == 0 else f"P_loc_{self._p_loc_dump_counter}.csv"
                    csv_path = os.path.join(self.debug_save_dir, csv_name)
                    pd.DataFrame(P_f_loc).to_csv(csv_path, index=False, header=False, float_format='%.8e')
                    print(f"Saved P_loc to {csv_path}")
                    self._p_loc_dump_counter += 1
            except Exception:
                pass
            det_S = np.linalg.det(S)
            print(f"Innovation covariance det: {det_S:.6e}")
        
        return xi_a, K_k
    
    def get_mean_and_spread(self, xi):
        mean = np.mean(xi, axis=1)
        stand_dev = np.std(xi, axis=1)
        return mean, stand_dev
    
    def print_covariance_info(self, xi_f, step_info=""):
        x_f, L_f = self.compute_mean_anomaly(xi_f)
        P_f = L_f @ L_f.T

        print(f"Ensemble size: {xi_f.shape[1]}")
        print(f"State dimension: {xi_f.shape[0]}")
    

        if self.localization_matrix is not None:
            P_f_loc = P_f * self.localization_matrix
            mid_point = self.M_dim // 2
        

def two_soliton_ic(x, a=[1.0, -0.8], x0=[0.0, 0.0], t0=-5.0):
    u0 = np.zeros_like(x)
    for i in range(len(a)):
        u0 += a[i] / np.cosh(a[i] * (x - x0[i] - a[i]**3 * t0))
    return u0

def get_model_path(exp=30, batch_size=64, m=1):
    base_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Unetmodels_multifinal_v2"
    filename = f"model_batch_{batch_size}_exp{exp}_k1_m{m}.pth"
    return os.path.join(base_dir, filename)

def adjust_ensemble_member(x_T, tau_i, alpha):
    return x_T + alpha * (tau_i - x_T)

def run_enkf_enhanced_unet_experiment(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                                      selected_members=None, alpha=1.0, n_prediction_ensemble=None, 
                                      save_data=True, print_covariance_info=False,
                                      localization_radius=None):

    np.random.seed(40)
    torch.manual_seed(40)
    random.seed(40)
    
    if n_prediction_ensemble is None:
        n_prediction_ensemble = N_ensemble
    
                                                       
    base_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\Covariance Localization\Unet effect"
    os.makedirs(base_dir, exist_ok=True)
    
                                                                             
                                                                  
    loc_tag = "" if localization_radius is None else f"_r{localization_radius:.2f}"
    exp_name = (
        f"N{N_ensemble}_Q{Q_d}_a{analysis_time:.1f}_l{lead_time:.1f}_np{n_prediction_ensemble}_a{alpha}"
        f"{loc_tag}"
    )
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
                                                     
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
        model_path = get_model_path(exp=30, batch_size=64, m=m)
        
        unet_model = UNetModel(k=1, m=m, m_max=4, kernel_size=3)
        unet_model.load_state_dict(torch.load(model_path, map_location=device))
        unet_model.to(device)
        unet_model.eval()
        
        u_min, u_max = np.min(u_truth), np.max(u_truth)
        ratio = int(dt_frame / dt_numerical)
        sqrt_ratio = np.sqrt(ratio)
        
        unet_system = UNetSystemK1(unet_model, device, m=m, u_min=u_min, u_max=u_max, dx=dx, dt_frame=dt_frame, nu=0.01)
                                                                              
        loc_r = localization_radius if localization_radius is not None else (L / 8.0)
        enkf_unet = EnKF_UNet(unet_system, N_ensemble, Q_std=sqrt_ratio * 5e-4, R_std=obs_noise_std, Q_d=Q_d,
                              localization_radius=loc_r, dx=dx)
        print(f"[ENKF-UNet] Analysis phase using localization radius: {loc_r}")
        if print_covariance_info:
            try:
                enkf_unet.debug_save_dir = exp_dir
            except Exception:
                pass
        
                             
        x0_est = np.zeros_like(u_truth[0])
        xi_ensemble = enkf_unet.initialize_enkf(x0_est, P0_std=0.1)
        
                                                              
        x_mean_analysis = np.zeros((analysis_idx + 1, M))
        std_analysis = np.zeros((analysis_idx + 1, M))
        mse_analysis = np.zeros(analysis_idx + 1)
        residual_analysis = np.zeros(analysis_idx + 1)
        spread_analysis = np.zeros(analysis_idx + 1)
        
                       
        x_mean, std = enkf_unet.get_mean_and_spread(xi_ensemble)
        x_mean_analysis[0] = x_mean
        std_analysis[0] = std
        mse_analysis[0] = np.mean((x_mean - u_truth[0])**2)
        residual_analysis[0] = np.mean(u_truth[0] - x_mean)
        spread_analysis[0] = np.mean(std)
        
                                                
        if 0 in observations:
            z_0 = observations[0]
                                                                                       
            xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_0, H, R, print_covariance=False)
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
                                                                                                    
                    print_cov = print_covariance_info and (frame_idx == 3)
                    xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_k, H, R, print_covariance=print_cov)
                
                                
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
        print(f"Data assimilation completed in {da_time:.2f}s")
        
                                     
        x_mean_da, std_da = enkf_unet.get_mean_and_spread(xi_ensemble)
        print(f"  DA final state: mean={np.mean(x_mean_da):.6f}, std={np.mean(std_da):.6f}")
        print(f"  DA xi_ensemble shape: {xi_ensemble.shape}, range: [{np.min(xi_ensemble):.6f}, {np.max(xi_ensemble):.6f}]")
        
                                                              
    else:
        print("Phase 1: Data Assimilation - Skipped (loaded from existing data)")
                                                  
        L = 50.0
        dx = 0.1
        dt_numerical = 1e-4
        dt_frame = 0.1
        analysis_idx = int(analysis_time / dt_numerical)
    
                              
    print(f"Running prediction phase from t={analysis_time}s to t={lead_time}s...")
    
                                            
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
    model_path = get_model_path(exp=30, batch_size=64, m=m)
    
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
                                                                           
    loc_r_pred = localization_radius if localization_radius is not None else (L / 8.0)
    enkf_pred = EnKF_UNet(unet_system, n_prediction_ensemble, Q_std=Q_std_pred, R_std=obs_noise_std, Q_d=Q_d,
                          localization_radius=loc_r_pred, dx=dx)
    print(f"[ENKF-UNet] Prediction phase using localization radius: {loc_r_pred}")
    if print_covariance_info:
        try:
            enkf_pred.debug_save_dir = exp_dir
        except Exception:
            pass
    
                                                                             
    print(f"Phase 2: Pure Prediction from t={analysis_time}s to t={lead_time}s")
    prediction_start_time = time.time()
    
    analysis_idx = int(analysis_time / dt_numerical)
    lead_idx = int(lead_time / dt_numerical)
    prediction_length = lead_idx - analysis_idx
    
    N_t_frame_pred = int((lead_time - analysis_time) / dt_frame)
    
    x_mean_prediction = np.zeros((prediction_length + 1, M))
    std_prediction = np.zeros((prediction_length + 1, M))
    mse_prediction = np.zeros(prediction_length + 1)
    residual_prediction = np.zeros(prediction_length + 1)
    spread_prediction = np.zeros(prediction_length + 1)
    
                                  
    x_mean, std = enkf_pred.get_mean_and_spread(xi_prediction)
    x_mean_prediction[0] = x_mean
    std_prediction[0] = std
    mse_prediction[0] = np.mean((x_mean - u_truth[analysis_idx])**2)
    residual_prediction[0] = np.mean(u_truth[analysis_idx] - x_mean)
    spread_prediction[0] = np.mean(std)
    
                                                                          
    print(f"  Initial prediction state: mean={np.mean(x_mean):.6f}, std={np.mean(std):.6f}, MSE={mse_prediction[0]:.6f}")
    print(f"  xi_prediction shape: {xi_prediction.shape}, range: [{np.min(xi_prediction):.6f}, {np.max(xi_prediction):.6f}]")
    if print_covariance_info:
        try:
                                                               
            x_f0, L_f0 = enkf_pred.compute_mean_anomaly(xi_prediction)
            P_f0 = L_f0 @ L_f0.T
            if enkf_pred.localization_matrix is not None:
                P_loc0 = P_f0 * enkf_pred.localization_matrix
            else:
                P_loc0 = P_f0
            if enkf_pred.debug_save_dir is not None:
                os.makedirs(enkf_pred.debug_save_dir, exist_ok=True)
                pd.DataFrame(P_loc0).to_csv(os.path.join(enkf_pred.debug_save_dir, 'P_loc_prediction_start.csv'),
                                            index=False, header=False, float_format='%.8e')
                print(f"Saved P_loc to {os.path.join(enkf_pred.debug_save_dir, 'P_loc_prediction_start.csv')}")
        except Exception:
            pass
    
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
    print(f"Prediction completed in {prediction_time:.2f}s")
    
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
    total_mean_history = np.zeros((total_length, M))
    total_std_history = np.zeros((total_length, M))
    total_mse_history = np.zeros(total_length)
    total_residual_history = np.zeros(total_length)
    total_spread_history = np.zeros(total_length)
    
                         
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
        'localization_radius': loc_r_pred,
        'da_time': da_time if not load_analysis else 0,
        'prediction_time': prediction_time,
        'da_save_dir': None,
        'exp_dir': exp_dir,
        'dt_numerical': dt_numerical,
        'dt_frame': dt_frame,
        'N_ensemble': N_ensemble,
        'n_prediction_ensemble': n_prediction_ensemble
    }

def run_enkf_unet_experiment(target_time, N_ensemble, Q_d):
                                          
    np.random.seed(40)
    torch.manual_seed(40)
    random.seed(40)
    
                       
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
                                 
    localization_radius = L / 8.0                                         
    enkf_unet = EnKF_UNet(unet_system, N_ensemble, Q_std= sqrt_ratio * 5e-4, R_std=obs_noise_std, Q_d=Q_d,
                         localization_radius=localization_radius, dx=dx)
    
                         
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
        xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_0, H, R, print_covariance=False)
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
                                                   
                print_cov = False                                                
                xi_ensemble, _ = enkf_unet.measurement_update(xi_ensemble, z_k, H, R, print_covariance=print_cov)

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
    alpha_actual = results['alpha']
    n_pred_actual = results['n_prediction_ensemble']
    
    time_vec = np.arange(len(mse)) * dt_numerical
    
                            
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    selected_idx_numerical = int(round(selected_time_s / dt_numerical))
    selected_idx_numerical = max(0, min(selected_idx_numerical, len(u_truth) - 1))
    
                                                                    
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
    
                           
    sample_step = 1000
    mask_upto = time_vec <= (selected_time_s + 1e-12)
    t_plot = time_vec[mask_upto][::sample_step]
    mse_plot = mse[mask_upto][::sample_step]
    residual_plot = residual_mean[mask_upto][::sample_step]
    spread_plot = spread[mask_upto][::sample_step]

                                                                     
    final_idx = min(selected_idx_numerical, len(mse) - 1)
    if t_plot.size == 0 or abs(t_plot[-1] - selected_time_s) > 1e-12:
        t_plot = np.append(t_plot, selected_time_s)
        mse_plot = np.append(mse_plot, mse[final_idx])
        residual_plot = np.append(residual_plot, residual_mean[final_idx])
        spread_plot = np.append(spread_plot, spread[final_idx])
    
                                       
    title_suffix = f"N{N_ensemble_actual}_alpha{alpha_actual}_npred{n_pred_actual}"
    
                         
    plt.figure(figsize=(12, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=2.5, label='Truth')
    plt.plot(x, selected_estimate, 'r--', linewidth=2.5, label='EnKF-UNet')
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF-UNet Prediction vs Truth at t = {selected_time_s:.2f}s (Analysis until t={analysis_time:.1f}s, α={alpha_actual}, N_pred={n_pred_actual})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f'prediction_vs_truth_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                            
    
                   
    plt.figure(figsize=(12, 6))
    plt.semilogy(t_plot, mse_plot, 'g-', linewidth=2.5, label='MSE')
    plt.axvline(x=analysis_time, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Analysis time (t={analysis_time:.1f}s)')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, label=f'Target time (t={selected_time_s:.2f}s)')
    plt.xlabel('Time t (s)', fontsize=14)
    plt.xlim(0.0, selected_time_s)
                          
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title(f'MSE Evolution: Data Assimilation + Pure Prediction (N_pred={n_pred_actual})', fontsize=16, fontweight='bold')
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
    plt.savefig(os.path.join(exp_dir, f'mse_evolution_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                            
    
                 
    plt.figure(figsize=(12, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=3, label='Truth')
    plt.plot(x, selected_estimate, 'r--', linewidth=2.5, label='EnKF-UNet')
    
    lower_2sigma = selected_estimate - 2*selected_std
    upper_2sigma = selected_estimate + 2*selected_std
    plt.fill_between(x, lower_2sigma, upper_2sigma, alpha=0.3, color='red', label='2σ bounds')
    
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF-UNet with Error Bands at t = {selected_time_s:.2f}s (Analysis until t={analysis_time:.1f}s, α={alpha_actual}, N_pred={n_pred_actual})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f'error_bands_t{selected_time_s:.1f}s_{title_suffix}.png'), dpi=300)
                            

                                                                              
    plt.figure(figsize=(14, 8))
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=5)
    if selected_idx_numerical <= analysis_idx:
        ensemble_snapshot = results['ensemble_final_da']
        phase_info = 'Data Assimilation'
    else:
        ensemble_snapshot = results['ensemble_final_pred']
        phase_info = 'Pure Prediction'
    num_members = ensemble_snapshot.shape[1]
    member_colors = plt.cm.viridis(np.linspace(0, 1, num_members))
    for j in range(num_members):
        plt.plot(x, ensemble_snapshot[:, j], color=member_colors[j], linewidth=1.2, alpha=0.6)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'All Ensemble Members vs Truth at t = {selected_time_s:.2f}s ({phase_info}) (α={alpha_actual}, N_pred={n_pred_actual})', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f'ensemble_members_vs_truth_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                            

                                                                                
    plt.figure(figsize=(14, 8))
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
                                                 
    if selected_idx_numerical <= analysis_idx:
        envelope_snapshot = results['ensemble_final_da']
        phase_info_env = 'Data Assimilation'
    else:
        envelope_snapshot = results['ensemble_final_pred']
        phase_info_env = 'Pure Prediction'
                                          
    lower_envelope = np.min(envelope_snapshot, axis=1)
    upper_envelope = np.max(envelope_snapshot, axis=1)
                                      
    plt.plot(x, selected_estimate, 'r--', linewidth=2.0, label='EnKF-UNet')
    plt.fill_between(x, lower_envelope, upper_envelope, color='tab:blue', alpha=0.2, label='Min–Max envelope')
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF-UNet with Min–Max Error Bands at t = {selected_time_s:.2f}s ({phase_info_env})' +
              f' (Analysis until t={analysis_time:.1f}s, α={alpha_actual}, N_pred={n_pred_actual})',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f'prediction_with_minmax_envelope_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                            
    
                       
    rmse_at_target = compute_rmse_over_domain(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical])
    
    rogue_threshold = 1.0
    rogue_region = u_truth[selected_idx_numerical] > rogue_threshold
    if not np.any(rogue_region):
        rogue_region = None
    peak_amp_error = compute_peak_amplitude_error(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical], rogue_region)
    mean_abs_err_peak_region = compute_mean_abs_error_region(u_truth[selected_idx_numerical], x_mean[selected_idx_numerical], rogue_region)
    
    print(f"RMSE (domain) at t={selected_time_s:.2f}s: {rmse_at_target:.6f}")
    if rogue_region is None:
        print(f"No rogue region detected")
    else:
        print(f"Mean absolute error in rogue region: {mean_abs_err_peak_region:.6f}")
    
                              
    analysis_idx_num = int(round(analysis_time / dt_numerical))
    analysis_idx_num = max(0, min(analysis_idx_num, len(u_truth) - 1))
    truth_at_analysis = u_truth[analysis_idx_num]
                                          
    est_at_analysis = results['x_mean_da'][analysis_idx_num] if 'x_mean_da' in results else x_mean[analysis_idx_num]
    std_at_analysis = results['std_da'][analysis_idx_num] if 'std_da' in results else std[analysis_idx_num]
    rmse_at_analysis = compute_rmse_over_domain(truth_at_analysis, est_at_analysis)
    rogue_region_ana = truth_at_analysis > rogue_threshold
    if not np.any(rogue_region_ana):
        rogue_region_ana = None
    peak_amp_err_ana = compute_peak_amplitude_error(truth_at_analysis, est_at_analysis, rogue_region_ana)
    mean_abs_err_peak_region_ana = compute_mean_abs_error_region(truth_at_analysis, est_at_analysis, rogue_region_ana)
    print(f"RMSE (domain) at analysis time t={analysis_time:.2f}s: {rmse_at_analysis:.6f}")
    
                                
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
        prediction_path = os.path.join(exp_dir, f'prediction_data_t{selected_time_s:.2f}s_{title_suffix}.csv')
        prediction_df.to_csv(prediction_path, index=False)
        
                                          
        metrics_data = {
            'target_time': [target_time],
            'analysis_time': [analysis_time],
            'lead_time': [lead_time],
            'selected_time': [selected_time_s],
            'N_ensemble': [N_ensemble_actual],
            'alpha': [alpha_actual],
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
        
                                          
        metrics_analysis = {
            'target_time': [target_time],
            'analysis_time': [analysis_time],
            'lead_time': [lead_time],
            'N_ensemble': [N_ensemble_actual],
            'alpha': [alpha_actual],
            'n_prediction_ensemble': [n_pred_actual],
            'Q_d': [Q_d],
            'rmse_domain_at_analysis': [rmse_at_analysis],
            'peak_amplitude_error_at_analysis': [peak_amp_err_ana],
            'mean_abs_error_peak_region_at_analysis': [mean_abs_err_peak_region_ana],
            'mse_at_analysis': [mse[analysis_idx_num] if len(mse) > analysis_idx_num else np.nan],
            'spread_at_analysis': [spread[analysis_idx_num] if len(spread) > analysis_idx_num else np.nan]
        }
        metrics_analysis_df = pd.DataFrame(metrics_analysis)
        metrics_analysis_path = os.path.join(exp_dir, f'metrics_analysis_time_{title_suffix}.csv')
        metrics_analysis_df.to_csv(metrics_analysis_path, index=False)
        
        print(f"Results saved to {exp_dir}")
    
    return results, exp_dir



def compare_alpha_npred_combinations_unet(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                                         alpha_values, n_prediction_values, selected_members=None, 
                                         save_data=True, time_chosen=None):
    base_dir = r"C:\\Users\\DELL\\Desktop\\Thesis code\\KdV code\\Numerical sol\\Final code\\Final_Final_code\\Data_assimilation\\Covariance Localization\\Unet results"
    os.makedirs(base_dir, exist_ok=True)
    
                                                                                           
    global _covariance_printed
    if '_covariance_printed' not in globals():
        _covariance_printed = False
    
                                                                    
    compare_name = f"cmp_N{N_ensemble}_Qd{Q_d}_a{analysis_time:.1f}_l{lead_time:.1f}"
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
            
                                                                 
            print_cov_info = not _covariance_printed
            if print_cov_info:
                _covariance_printed = True
                print(f"    [Covariance localization info will be printed for this experiment]")
            
                                                                                 
            results = run_enkf_enhanced_unet_experiment(
                target_time, analysis_time, lead_time, N_ensemble, Q_d,
                selected_members, alpha, n_pred, save_data=True, print_covariance_info=print_cov_info
            )
            
            key = (alpha, n_pred)
            results_dict[key] = results
            
                                                                                                 
            exp_dir = results.get('exp_dir', compare_dir)
            save_localization_heatmaps(results, exp_dir, alpha=alpha, n_pred=n_pred)

                                                                      
            try:
                time_vec = np.arange(len(results['mse_total'])) * results['dt_numerical']
                ts_df = pd.DataFrame({
                    'time': time_vec,
                    'mse': results['mse_total'],
                    'residual_mean': results['residual_mean_total'],
                    'spread': results['spread_total']
                })
                os.makedirs(exp_dir, exist_ok=True)
                ts_df.to_csv(os.path.join(exp_dir, f'time_series_alpha{alpha}_npred{n_pred}.csv'), index=False)
            except Exception:
                pass

                                             
            selected_idx = int(round(selected_time_s / results['dt_numerical']))
            selected_idx = max(0, min(selected_idx, len(results['u_truth']) - 1))
            rmse = compute_rmse_over_domain(results['u_truth'][selected_idx], results['x_mean_total'][selected_idx])
            rmse_grid[i, j] = rmse
            
                                                                                         
            try:
                x = results['x']
                dt_numerical = results['dt_numerical']
                analysis_idx = results['analysis_idx']
                u_truth = results['u_truth']
                x_mean_total = results['x_mean_total']
                std_total = results['std_total']
                ensemble_da = results['ensemble_final_da']
                ensemble_pred = results['ensemble_final_pred']
                N_ensemble_actual = results['N_ensemble']
                alpha_actual = results['alpha']
                n_pred_actual = results['n_prediction_ensemble']
                loc_val = results.get('localization_radius')
                loc_str = f"_r{loc_val:.2f}" if isinstance(loc_val, (int, float)) else ""
                title_suffix = f"N{N_ensemble_actual}_alpha{alpha_actual}_npred{n_pred_actual}{loc_str}"

                selected_idx = int(round(selected_time_s / dt_numerical))
                selected_idx = max(0, min(selected_idx, len(u_truth) - 1))
                truth_sel = u_truth[selected_idx]
                est_sel = x_mean_total[selected_idx]
                std_sel = std_total[selected_idx]

                                        
                plt.figure(figsize=(12, 6))
                plt.plot(x, truth_sel, 'k-', lw=2.5, label='Truth')
                plt.plot(x, est_sel, 'r--', lw=2.5, label='EnKF-UNet')
                plt.xlabel('Spatial Position x')
                plt.ylabel('Wave Amplitude u')
                plt.title(f'Prediction vs Truth at t = {selected_time_s:.2f}s')
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'prediction_vs_truth_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                  
                time_vec = np.arange(len(results['mse_total'])) * dt_numerical
                plt.figure(figsize=(12, 6))
                plt.semilogy(time_vec, results['mse_total'], 'g-', lw=2.5, label='MSE')
                plt.axvline(x=analysis_time, color='blue', ls='--', lw=2, alpha=0.7, label=f'Analysis t={analysis_time:.1f}s')
                plt.axvline(x=selected_time_s, color='red', ls='--', lw=2, label=f'Target t={selected_time_s:.2f}s')
                plt.xlabel('Time t (s)'); plt.ylabel('MSE'); plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'mse_evolution_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                      
                plt.figure(figsize=(12, 6))
                plt.plot(x, truth_sel, 'k-', lw=3, label='Truth')
                plt.plot(x, est_sel, 'r--', lw=2.5, label='EnKF-UNet')
                lower_2sigma = est_sel - 2*std_sel
                upper_2sigma = est_sel + 2*std_sel
                plt.fill_between(x, lower_2sigma, upper_2sigma, alpha=0.3, color='red', label='2σ bounds')
                plt.xlabel('Spatial Position x'); plt.ylabel('Wave Amplitude u')
                plt.title(f'Error Bands at t = {selected_time_s:.2f}s')
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'error_bands_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                                          
                if selected_idx <= analysis_idx:
                    ensemble_snapshot = ensemble_da
                    phase_info = 'Data Assimilation'
                else:
                    ensemble_snapshot = ensemble_pred
                    phase_info = 'Pure Prediction'

                                              
                plt.figure(figsize=(14, 8))
                plt.plot(x, truth_sel, 'k-', lw=3, label='Truth', zorder=5)
                num_members = ensemble_snapshot.shape[1]
                member_colors = plt.cm.viridis(np.linspace(0, 1, num_members))
                for j2 in range(num_members):
                    plt.plot(x, ensemble_snapshot[:, j2], color=member_colors[j2], lw=1.2, alpha=0.6)
                plt.xlabel('Spatial Position x'); plt.ylabel('Wave Amplitude u')
                plt.title(f'All Ensemble Members vs Truth at t = {selected_time_s:.2f}s ({phase_info})')
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'ensemble_members_vs_truth_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                                    
                plt.figure(figsize=(14, 8))
                plt.plot(x, truth_sel, 'k-', lw=3, label='Truth', zorder=10)
                lower_envelope = np.min(ensemble_snapshot, axis=1)
                upper_envelope = np.max(ensemble_snapshot, axis=1)
                plt.plot(x, est_sel, 'r--', lw=2.0, label='EnKF-UNet')
                plt.fill_between(x, lower_envelope, upper_envelope, color='tab:blue', alpha=0.2, label='Min–Max envelope')
                plt.xlabel('Spatial Position x'); plt.ylabel('Wave Amplitude u')
                plt.title(f'Prediction with Min–Max Envelope at t = {selected_time_s:.2f}s ({phase_info})')
                plt.legend(loc='upper right'); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'prediction_with_minmax_envelope_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                             
                rmse_at_target = compute_rmse_over_domain(truth_sel, est_sel)
                rogue_threshold = 1.0
                rogue_region = truth_sel > rogue_threshold
                if not np.any(rogue_region):
                    rogue_region = None
                peak_amp_error = compute_peak_amplitude_error(truth_sel, est_sel, rogue_region)
                mean_abs_err_peak_region = compute_mean_abs_error_region(truth_sel, est_sel, rogue_region)
                metrics_df = pd.DataFrame({
                    'target_time': [target_time],
                    'analysis_time': [analysis_time],
                    'lead_time': [lead_time],
                    'selected_time': [selected_time_s],
                    'N_ensemble': [N_ensemble_actual],
                    'alpha': [alpha_actual],
                    'n_prediction_ensemble': [n_pred_actual],
                    'Q_d': [Q_d],
                    'rmse_domain': [rmse_at_target],
                    'peak_amplitude_error': [peak_amp_error],
                    'mean_abs_error_peak_region': [mean_abs_err_peak_region],
                    'final_mse': [results['mse_total'][-1] if len(results['mse_total'])>0 else np.nan],
                    'final_spread': [results['spread_total'][-1] if len(results['spread_total'])>0 else np.nan]
                })
                metrics_df.to_csv(os.path.join(exp_dir, f'metrics_{title_suffix}.csv'), index=False)
            except Exception:
                pass
    
    print("\nGenerating combined comparison plots...")
    
                  
    plt.figure(figsize=(10, 7))
    im = plt.imshow(rmse_grid, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='RMSE')
    
    plt.xticks(range(len(n_prediction_values)), n_prediction_values)
    plt.yticks(range(len(alpha_values)), alpha_values)
    plt.xlabel('N_prediction_ensemble', fontsize=14)
    plt.ylabel('Alpha (α)', fontsize=14)
    plt.title(f'RMSE Heatmap at t = {selected_time_s:.2f}s', fontsize=16, fontweight='bold')
    
                          
    for i in range(len(alpha_values)):
        for j in range(len(n_prediction_values)):
            plt.text(j, i, f'{rmse_grid[i, j]:.4f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'rmse_heatmap_combinations_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
    print(f"Combination comparison results saved to {compare_dir}")
    return results_dict, compare_dir

def test_covariance_localization():
                                          
    np.random.seed(40)
    torch.manual_seed(40)
    random.seed(40)
    
    print("=== Testing Covariance Localization ===")
    
                     
    M = 100                         
    dx = 0.1                
    L = M * dx                 
    localization_radius = L / 8.0                       
    
    print(f"Domain: M={M}, dx={dx}, L={L}")
    print(f"Localization radius: {localization_radius}")
    
                                
    rho = create_localization_matrix(M, dx, localization_radius)
    
    print(f"\nLocalization matrix shape: {rho.shape}")
    print(f"Localization matrix range: [{np.min(rho):.6f}, {np.max(rho):.6f}]")
    
                              
    print(f"\nPeriodic boundary test:")
    print(f"rho[0,1] = {rho[0,1]:.6f}")
    print(f"rho[0,-1] = {rho[0,-1]:.6f} (should be similar for periodic domain)")
    print(f"rho[1,0] = {rho[1,0]:.6f}")
    print(f"rho[-1,0] = {rho[-1,0]:.6f} (should be similar for periodic domain)")
    
    
                                                                                                     
    try:
        import matplotlib.pyplot as plt

                                                   
        plt.figure(figsize=(8, 6))
        im = plt.imshow(rho, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Localization weight')
        plt.title('Localization matrix (rho)')
        plt.xlabel('j')
        plt.ylabel('i')
        plt.tight_layout()
        save_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\Covariance Localization\Unet results"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'localization_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

                                                                                              
                                                   
        N = 50                           
        rng = np.random.default_rng(42)
        xi = rng.normal(0.0, 1.0, (M, N))
                                                                                                     
        xi = (xi + np.roll(xi, 1, axis=0) + np.roll(xi, -1, axis=0)) / 3.0
        x_mean = np.mean(xi, axis=1)
        L_f = (xi - x_mean[:, None]) / np.sqrt(max(N - 1, 1))
        P_f = L_f @ L_f.T

        vmin = np.min(P_f)
        vmax = np.max(P_f)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(P_f, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Covariance')
        plt.title('Ensemble covariance matrix (P_f)')
        plt.xlabel('j')
        plt.ylabel('i')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ensemble_covariance_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

                                 
        P_f_loc = P_f * rho
        plt.figure(figsize=(8, 6))
        im = plt.imshow(P_f_loc, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Covariance (localized)')
        plt.title('Ensemble covariance after localization (P_f ∘ rho)')
        plt.xlabel('j')
        plt.ylabel('i')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ensemble_covariance_after_localization.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved heatmaps to: {save_dir}")

    except ImportError:
        print("Matplotlib not available for plotting")
    
    print("=== Covariance Localization Test Complete ===")
    return rho

def save_localization_heatmaps(results, save_dir, alpha=None, n_pred=None, localization_radius=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return

    x = results.get('x', None)
    xi = results.get('ensemble_final_da', None)
    if xi is None:
        xi = results.get('ensemble_final_pred', None)
    if x is None or xi is None:
                                                  
        return

    M = len(x)
    if M < 2:
                                        
        return
    dx = float(x[1] - x[0])
    L = float(x[-1] - x[0] + dx)
                                                                  
    if localization_radius is None:
        localization_radius = results.get('localization_radius', L / 8.0)

                                    
    rho = create_localization_matrix(M, dx, localization_radius)

                                      
    N = xi.shape[1]
    x_mean = np.mean(xi, axis=1)
    L_f = (xi - x_mean[:, None]) / np.sqrt(max(N - 1, 1))
    P_f = L_f @ L_f.T
    P_f_loc = P_f * rho

    tag = ""
    if alpha is not None:
        tag += f"_alpha{alpha}"
    if n_pred is not None:
        tag += f"_npred{n_pred}"
    if localization_radius is not None:
        tag += f"_locR{localization_radius:.3f}"

    os.makedirs(save_dir, exist_ok=True)

            
    plt.figure(figsize=(8, 6))
    im = plt.imshow(rho, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Localization weight')
    plt.title('Localization matrix (rho)')
    plt.xlabel('j')
    plt.ylabel('i')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rho{tag}.png'), dpi=300, bbox_inches='tight')
    plt.close()

                                     
    vmin = float(np.min(P_f))
    vmax = float(np.max(P_f))

            
    plt.figure(figsize=(8, 6))
    im = plt.imshow(P_f, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Covariance')
    plt.title('Ensemble covariance matrix (P_f)')
    plt.xlabel('j')
    plt.ylabel('i')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Pf{tag}.png'), dpi=300, bbox_inches='tight')
    plt.close()

                      
    plt.figure(figsize=(8, 6))
    im = plt.imshow(P_f_loc, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Covariance (localized)')
    plt.title('Ensemble covariance after localization (P_f ∘ rho)')
    plt.xlabel('j')
    plt.ylabel('i')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Pf_loc{tag}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_radius_npred_unet(target_time, analysis_time, lead_time, N_ensemble, Q_d,
                               localization_radii, n_prediction_values, selected_members=None,
                               save_data=True, time_chosen=None):
                                   
    base_dir = r"C:\\Users\\DELL\\Desktop\\Thesis code\\KdV code\\Numerical sol\\Final code\\Final_Final_code\\Data_assimilation\\Covariance Localization\\Unet effect"
    os.makedirs(base_dir, exist_ok=True)

    compare_name = f"cmpR_N{N_ensemble}_Qd{Q_d}_a{analysis_time:.1f}_l{lead_time:.1f}"
    compare_dir = os.path.join(base_dir, compare_name)
    os.makedirs(compare_dir, exist_ok=True)

    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)

    rmse_grid = np.zeros((len(localization_radii), len(n_prediction_values)))
    metrics_rows = []

    total = len(localization_radii) * len(n_prediction_values)
    idx = 0
    for i, loc_r in enumerate(localization_radii):
        for j, n_pred in enumerate(n_prediction_values):
            idx += 1
            print(f"\nRunning combo {idx}/{total}: locR={loc_r}, n_pred={n_pred}, alpha fixed to 1.0")

            results = run_enkf_enhanced_unet_experiment(
                target_time, analysis_time, lead_time, N_ensemble, Q_d,
                selected_members=selected_members, alpha=1.0, n_prediction_ensemble=n_pred,
                save_data=True, print_covariance_info=(idx == 1), localization_radius=loc_r
            )

            exp_dir = results.get('exp_dir', compare_dir)
            save_localization_heatmaps(results, exp_dir, alpha=1.0, n_pred=n_pred, localization_radius=loc_r)

                                                                                          
            try:
                x = results['x']
                dt_numerical = results['dt_numerical']
                analysis_idx_res = results['analysis_idx']
                u_truth_res = results['u_truth']
                x_mean_total = results['x_mean_total']
                std_total = results['std_total']
                ensemble_da = results['ensemble_final_da']
                ensemble_pred = results['ensemble_final_pred']
                N_ensemble_actual = results['N_ensemble']
                alpha_actual = results['alpha']
                n_pred_actual = results['n_prediction_ensemble']
                title_suffix = f"N{N_ensemble_actual}_alpha{alpha_actual}_npred{n_pred_actual}"

                selected_idx_plot = int(round(selected_time_s / dt_numerical))
                selected_idx_plot = max(0, min(selected_idx_plot, len(u_truth_res) - 1))
                truth_sel = u_truth_res[selected_idx_plot]
                est_sel = x_mean_total[selected_idx_plot]
                std_sel = std_total[selected_idx_plot]

                                        
                plt.figure(figsize=(12, 6))
                plt.plot(x, truth_sel, 'k-', lw=2.5, label='Truth')
                plt.plot(x, est_sel, 'r--', lw=2.5, label='EnKF-UNet')
                plt.xlabel('Spatial Position x')
                plt.ylabel('Wave Amplitude u')
                plt.title(f'Prediction vs Truth at t = {selected_time_s:.2f}s')
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'prediction_vs_truth_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                  
                time_vec_plot = np.arange(len(results['mse_total'])) * dt_numerical
                plt.figure(figsize=(12, 6))
                plt.semilogy(time_vec_plot, results['mse_total'], 'g-', lw=2.5, label='MSE')
                plt.axvline(x=analysis_time, color='blue', ls='--', lw=2, alpha=0.7, label=f'Analysis t={analysis_time:.1f}s')
                plt.axvline(x=selected_time_s, color='red', ls='--', lw=2, label=f'Target t={selected_time_s:.2f}s')
                plt.xlabel('Time t (s)'); plt.ylabel('MSE'); plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'mse_evolution_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                      
                plt.figure(figsize=(12, 6))
                plt.plot(x, truth_sel, 'k-', lw=3, label='Truth')
                plt.plot(x, est_sel, 'r--', lw=2.5, label='EnKF-UNet')
                lower_2sigma = est_sel - 2*std_sel
                upper_2sigma = est_sel + 2*std_sel
                plt.fill_between(x, lower_2sigma, upper_2sigma, alpha=0.3, color='red', label='2σ bounds')
                plt.xlabel('Spatial Position x'); plt.ylabel('Wave Amplitude u')
                plt.title(f'Error Bands at t = {selected_time_s:.2f}s')
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'error_bands_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                                          
                if selected_idx_plot <= analysis_idx_res:
                    ensemble_snapshot = ensemble_da
                    phase_info = 'Data Assimilation'
                else:
                    ensemble_snapshot = ensemble_pred
                    phase_info = 'Pure Prediction'

                                              
                plt.figure(figsize=(14, 8))
                plt.plot(x, truth_sel, 'k-', lw=3, label='Truth', zorder=5)
                num_members = ensemble_snapshot.shape[1]
                member_colors = plt.cm.viridis(np.linspace(0, 1, num_members))
                for j2 in range(num_members):
                    plt.plot(x, ensemble_snapshot[:, j2], color=member_colors[j2], lw=1.2, alpha=0.6)
                plt.xlabel('Spatial Position x'); plt.ylabel('Wave Amplitude u')
                plt.title(f'All Ensemble Members vs Truth at t = {selected_time_s:.2f}s ({phase_info})')
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'ensemble_members_vs_truth_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                                    
                plt.figure(figsize=(14, 8))
                plt.plot(x, truth_sel, 'k-', lw=3, label='Truth', zorder=10)
                lower_envelope = np.min(ensemble_snapshot, axis=1)
                upper_envelope = np.max(ensemble_snapshot, axis=1)
                plt.plot(x, est_sel, 'r--', lw=2.0, label='EnKF-UNet')
                plt.fill_between(x, lower_envelope, upper_envelope, color='tab:blue', alpha=0.2, label='Min–Max envelope')
                plt.xlabel('Spatial Position x'); plt.ylabel('Wave Amplitude u')
                plt.title(f'Prediction with Min–Max Envelope at t = {selected_time_s:.2f}s ({phase_info})')
                plt.legend(loc='upper right'); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(exp_dir, f'prediction_with_minmax_envelope_t{selected_time_s:.2f}s_{title_suffix}.png'), dpi=300)
                plt.close()

                                                                                
                rmse_at_target = compute_rmse_over_domain(truth_sel, est_sel)
                rogue_threshold = 1.0
                rogue_region = truth_sel > rogue_threshold
                if not np.any(rogue_region):
                    rogue_region = None
                peak_amp_error = compute_peak_amplitude_error(truth_sel, est_sel, rogue_region)
                mean_abs_err_peak_region = compute_mean_abs_error_region(truth_sel, est_sel, rogue_region)
                metrics_df = pd.DataFrame({
                    'target_time': [target_time],
                    'analysis_time': [analysis_time],
                    'lead_time': [lead_time],
                    'selected_time': [selected_time_s],
                    'N_ensemble': [N_ensemble_actual],
                    'alpha': [alpha_actual],
                    'n_prediction_ensemble': [n_pred_actual],
                    'Q_d': [Q_d],
                    'localization_radius': [loc_r],
                    'rmse_domain': [rmse_at_target],
                    'peak_amplitude_error': [peak_amp_error],
                    'mean_abs_error_peak_region': [mean_abs_err_peak_region],
                    'final_mse': [results['mse_total'][-1] if len(results['mse_total'])>0 else np.nan],
                    'final_spread': [results['spread_total'][-1] if len(results['spread_total'])>0 else np.nan]
                })
                metrics_df.to_csv(os.path.join(exp_dir, f'metrics_{title_suffix}.csv'), index=False)
            except Exception:
                pass

                                                                                                                            
            try:
                dt_num = results['dt_numerical']
                total_len = len(results['mse_total'])
                time_vec = np.arange(total_len) * dt_num
                mse_full = np.asarray(results['mse_total'], dtype=float)
                resid_full = np.asarray(results.get('residual_mean_total', np.full(total_len, np.nan)), dtype=float)
                spread_full = np.asarray(results.get('spread_total', np.full(total_len, np.nan)), dtype=float)
                                                                                        
                x_mean_total = np.asarray(results['x_mean_total'])
                missing_mask = ~np.any(x_mean_total, axis=1)
                                                                    
                mse_full[missing_mask] = np.nan
                resid_full[missing_mask] = np.nan
                spread_full[missing_mask] = np.nan
                                                            
                mse_full = pd.Series(mse_full).ffill().bfill().values
                resid_full = pd.Series(resid_full).ffill().bfill().values
                spread_full = pd.Series(spread_full).ffill().bfill().values
                ts_df = pd.DataFrame({
                    'time': time_vec,
                    'mse': mse_full,
                    'residual_mean': resid_full,
                    'spread': spread_full
                })
                os.makedirs(exp_dir, exist_ok=True)
                ts_name = f'time_series_R{loc_r:.3f}_npred{n_pred}.csv'
                ts_df.to_csv(os.path.join(exp_dir, ts_name), index=False)
            except Exception:
                pass

                                      
            selected_idx = int(round(selected_time_s / results['dt_numerical']))
            selected_idx = max(0, min(selected_idx, len(results['u_truth']) - 1))
            truth_sel = results['u_truth'][selected_idx]
            est_sel = results['x_mean_total'][selected_idx]
            rmse = compute_rmse_over_domain(truth_sel, est_sel)
            rmse_grid[i, j] = rmse

            rogue_threshold = 1.0
            rogue_region = truth_sel > rogue_threshold
            if not np.any(rogue_region):
                rogue_region = None
            peak_amp_error = compute_peak_amplitude_error(truth_sel, est_sel, rogue_region)
            mean_abs_err_peak_region = compute_mean_abs_error_region(truth_sel, est_sel, rogue_region)

            analysis_idx_full = int(round(analysis_time / results['dt_numerical']))
            analysis_idx_full = max(0, min(analysis_idx_full, len(results['u_truth']) - 1))
            truth_analysis = results['u_truth'][analysis_idx_full]
            est_analysis = results['x_mean_total'][analysis_idx_full]
            rmse_analysis = compute_rmse_over_domain(truth_analysis, est_analysis)
            rogue_region_analysis = truth_analysis > rogue_threshold
            if not np.any(rogue_region_analysis):
                rogue_region_analysis = None
            peak_amp_error_analysis = compute_peak_amplitude_error(truth_analysis, est_analysis, rogue_region_analysis)
            mean_abs_err_peak_analysis = compute_mean_abs_error_region(truth_analysis, est_analysis, rogue_region_analysis)

            metrics_rows.append({
                'target_time': target_time,
                'analysis_time': analysis_time,
                'lead_time': lead_time,
                'selected_time': selected_time_s,
                'N_ensemble': results['N_ensemble'],
                'alpha': 1.0,
                'n_prediction_ensemble': results['n_prediction_ensemble'],
                'Q_d': Q_d,
                'localization_radius': loc_r,
                'rmse_domain': rmse,
                'peak_amplitude_error': peak_amp_error,
                'mean_abs_error_peak_region': mean_abs_err_peak_region,
                'final_mse': results['mse_total'][-1] if len(results['mse_total'])>0 else np.nan,
                'final_spread': results['spread_total'][-1] if len(results['spread_total'])>0 else np.nan,
                                          
                'rmse_domain_at_analysis': rmse_analysis,
                'peak_amplitude_error_at_analysis': peak_amp_error_analysis,
                'mean_abs_error_peak_region_at_analysis': mean_abs_err_peak_analysis,
                'mse_at_analysis': results['mse_total'][int(round(analysis_time / results['dt_numerical']))] if len(results['mse_total']) > int(round(analysis_time / results['dt_numerical'])) else np.nan,
                'spread_at_analysis': results['spread_total'][int(round(analysis_time / results['dt_numerical']))] if len(results['spread_total']) > int(round(analysis_time / results['dt_numerical'])) else np.nan
            })

                             
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(compare_dir, 'metrics_radius_npred.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

                                               
                                                                                        
    rmse_grid_from_csv = np.full((len(localization_radii), len(n_prediction_values)), np.nan)
    for i, loc_r in enumerate(localization_radii):
        for j, n_pred in enumerate(n_prediction_values):
            match = metrics_df[(np.isclose(metrics_df['localization_radius'], loc_r)) &
                               (metrics_df['n_prediction_ensemble'] == n_pred)]
            if len(match) > 0:
                rmse_grid_from_csv[i, j] = float(match['rmse_domain'].values[-1])

                                                                  
    if np.isnan(rmse_grid_from_csv).any() and not np.isnan(rmse_grid).all():
        rmse_grid_plot = rmse_grid
    else:
        rmse_grid_plot = rmse_grid_from_csv

                                                          
    try:
        pd.DataFrame(rmse_grid_plot, index=[f"{r:.4f}" for r in localization_radii], columns=n_prediction_values)\
            .to_csv(os.path.join(compare_dir, f'rmse_heatmap_values_t{selected_time_s:.2f}.csv'))
    except Exception:
        pass

    plt.figure(figsize=(12, 8))
    im = plt.imshow(rmse_grid_plot, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='RMSE (domain)')
    plt.xticks(range(len(n_prediction_values)), n_prediction_values)
                           
    try:
        Ldomain = 50.0                                             
        def _fmt_label(r):
            if isinstance(r, (int, float)) and r > 0:
                denom = Ldomain / float(r)
                if abs(denom - round(denom)) < 1e-6:
                    return f"L/{int(round(denom))}"
                return f"L/{denom:.2f}"
            return str(r)
        y_labels = [_fmt_label(r) for r in localization_radii]
    except Exception:
        y_labels = [f"{r:.2f}" for r in localization_radii]
    plt.yticks(range(len(localization_radii)), y_labels)
    plt.xlabel('N_prediction_ensemble', fontsize=14)
    plt.ylabel('Localization radius (L/xx)', fontsize=14)
    plt.title(f'RMSE Heatmap (alpha=1) at t = {selected_time_s:.2f}s', fontsize=16, fontweight='bold')

    for i in range(len(localization_radii)):
        for j in range(len(n_prediction_values)):
            plt.text(j, i, f'{rmse_grid_plot[i, j]:.4f}', ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'heatmap_rmse_r_vs_np_t{selected_time_s:.2f}.png'), dpi=300, bbox_inches='tight')

    print(f"Radius-n_pred comparison saved to {compare_dir}")
    return metrics_df, compare_dir


def summarize_localization_radius_metrics(metrics_df, compare_dir, n_pred_filter=None):
    """
    Summarize RMSE- and peak-related metrics as a function of localization radius,
    print them for quick inspection, and save a RMSE-vs-radius line plot.
    """
    if metrics_df is None or metrics_df.empty:
        print("No metrics available to summarize localization radius effects.")
        return

    summary_df = metrics_df.copy()
    if n_pred_filter is not None:
        filtered = summary_df[summary_df['n_prediction_ensemble'] == n_pred_filter]
        if not filtered.empty:
            summary_df = filtered

    required_cols = [
        'rmse_domain_at_analysis',
        'peak_amplitude_error_at_analysis',
        'mean_abs_error_peak_region_at_analysis'
    ]
    missing_cols = [c for c in required_cols if c not in summary_df.columns]
    if missing_cols:
        print(f"Warning: missing analysis-time columns {missing_cols}; skipping summary.")
        return

    summary = summary_df.groupby('localization_radius', as_index=False)[required_cols].mean()
    summary = summary.sort_values('localization_radius')

    print("\n=== Metrics vs localization radius ===")
    for row in summary.itertuples(index=False):
        radius = row.localization_radius
        rmse = row.rmse_domain_at_analysis
        peak_err = row.peak_amplitude_error_at_analysis
        mean_abs_peak = row.mean_abs_error_peak_region_at_analysis
        print(f"radius={radius:.6f} -> RMSE@analysis={rmse:.6f}, Peak error@analysis={peak_err:.6f}, Mean abs.@analysis={mean_abs_peak:.6f}")

    os.makedirs(compare_dir, exist_ok=True)
    summary_csv = os.path.join(compare_dir, 'localization_radius_summary.csv')
    summary.to_csv(summary_csv, index=False)

    analysis_time_val = None
    if 'analysis_time' in summary_df.columns and not summary_df['analysis_time'].isna().all():
        analysis_time_val = float(summary_df['analysis_time'].iloc[0])

    plt.figure(figsize=(10, 6))
    plt.plot(summary['localization_radius'], summary['rmse_domain_at_analysis'],
             'o-', linewidth=2, markersize=6, color='tab:blue')
    plt.xlabel('Localization radius', fontsize=12)
    plt.ylabel('RMSE at analysis time', fontsize=12)
    if analysis_time_val is not None:
        title_str = f'RMSE (analysis t={analysis_time_val:.2f}) vs Localization Radius'
    else:
        title_str = 'RMSE vs Localization Radius (analysis time)'
    plt.title(title_str, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(compare_dir, 'rmse_vs_localization_radius.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved localization-radius summary CSV to: {summary_csv}")
    print(f"Saved RMSE vs localization radius plot to: {plot_path}")

if __name__ == "__main__":        

    L = 50.0
    localization_radii = [L/100.0, L/80.0, L/60.0, L/50.0, L/40.0, L/20.0, L/10.0, L/8.0, L/6.0, L/4.0, L/2.0, L/1.0]
    n_prediction_values = [150]

    metrics_df, compare_dir_combo = compare_radius_npred_unet(
        target_time=15.0,
        analysis_time=6.0,
        lead_time=7.8,
        N_ensemble=150,
        Q_d=10,
        localization_radii=localization_radii,
        n_prediction_values=n_prediction_values,
        selected_members=None,
        save_data=True,
        time_chosen=7.8
    )

    print(f"Results saved to: {compare_dir_combo}")

    summary_n_pred = n_prediction_values[0] if len(n_prediction_values) == 1 else None
    summarize_localization_radius_metrics(metrics_df, compare_dir_combo, n_pred_filter=summary_n_pred)

