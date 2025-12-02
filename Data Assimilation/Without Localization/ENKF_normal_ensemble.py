import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from scipy.linalg import sqrtm
import random

                                                                

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

def adjust_ensemble_member(x_T, tau_i, alpha):

    return x_T + alpha * (tau_i - x_T)

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
        
        if is_1d:
            return F_ensemble.flatten()
        else:
            return F_ensemble
    
    def M_op(self, xi_ensemble):
        k1 = self.F(xi_ensemble)
        k2 = self.F(xi_ensemble + 0.5 * self.dt * k1)
        k3 = self.F(xi_ensemble + 0.5 * self.dt * k2)
        k4 = self.F(xi_ensemble + self.dt * k3)
        xi_new = xi_ensemble + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
                                            
        return xi_new

class EnKF:   
    def __init__(self, system, N, Q_std, R_std, Q_d):
        self.system = system
        self.M_dim = system.M  
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
                                                                         
        w = self.L_Q @ eta
        return w
    
    def initialize_enkf(self, x0, P0_std=0.1):
        xi_ensemble = np.zeros((self.M_dim, self.N))
        
        
        perturbation = np.random.normal(0, P0_std, (self.M_dim, self.N))
        xi_ensemble= x0[:, np.newaxis] + perturbation
                                                                     
        
        return xi_ensemble
    
    def time_update(self, xi_a):
        
        w_k = self.generate_process_noise()
        xi_f = self.system.M_op(xi_a) + self.G @ w_k
                                                     
        
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


def run_enkf_enhanced_experiment(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                                selected_members=None, alpha=1.0, n_prediction_ensemble=None,
                                save_data=True):

    L = 50.0
    dx = 0.1
    M = int(L / dx)
    dt = 1e-4
    T = max(target_time, lead_time) + 0.1
    N_t = int(T / dt)
    x = np.linspace(-L/2, L/2, M, endpoint=False)
    
    analysis_idx = int(analysis_time / dt)
    lead_idx = int(lead_time / dt)
    target_idx = int(target_time / dt)
    
    print(f"\n--- Running Enhanced EnKF with N = {N_ensemble} ---")
    print(f"Analysis time: {analysis_time:.2f}s (idx: {analysis_idx})")
    print(f"Lead time: {lead_time:.2f}s (idx: {lead_idx})")
    print(f"Target time: {target_time:.2f}s (idx: {target_idx})")
    print(f"System: M={M}, dx={dx:.3f}, dt={dt}")
    
    system = mKdVSystem(M, dx, dt)
    
                                     
    truth_filename = f"u_truth_T{T:.1f}_dt{dt}_M{M}_dx{dx:.3f}.csv"
    truth_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_newresult\truth_data"
    os.makedirs(truth_dir, exist_ok=True)
    truth_path = os.path.join(truth_dir, truth_filename)
    
    if os.path.exists(truth_path):
        print(f"Loading existing truth solution: {truth_filename}")
        u_truth = pd.read_csv(truth_path, header=None).values
        required_rows = N_t + 1
        if u_truth.shape[1] == M and u_truth.shape[0] >= required_rows:
            u_truth = u_truth[:required_rows, :]
            print(f"Successfully loaded truth solution with shape {u_truth.shape}")
        else:
            print(f"Existing file has incompatible dimensions. Expected: ({required_rows}, {M}), Got: {u_truth.shape}")
            u_truth = None
    else:
        u_truth = None
    
    if u_truth is None:
        print("Generating new truth solution...")
        start_time = time.time()
        u0_truth = two_soliton_ic(x)
        u_truth = np.zeros((N_t + 1, M))
        u_truth[0] = u0_truth
        
        u = u0_truth.copy()
        for k in range(N_t):
            if (k+1) % 10000 == 0:
                progress = (k+1)/N_t*100
                elapsed = time.time() - start_time
                eta = elapsed / (k+1) * (N_t - k - 1)
                print(f"  Progress: {progress:.1f}%, ETA: {eta:.1f}s")
            u = system.M_op(u)
            u_truth[k+1] = u
        
        generation_time = time.time() - start_time
        print(f"Truth solution generated in {generation_time:.2f}s")
        
                                            
        pd.DataFrame(u_truth).to_csv(truth_path, index=False, header=False, float_format='%.8e')
        print(f"Truth solution saved to: {truth_path}")
    
                       
    obs_ratio = 0.1
    obs_interval = 3000
    obs_noise_std = 0.1
    
    n_obs = int(obs_ratio * M)
    obs_indices = np.sort(np.random.choice(M, n_obs, replace=False))
    
    H = np.zeros((n_obs, M))
    for i, idx in enumerate(obs_indices):
        H[i, idx] = 1.0
    R = (obs_noise_std**2) * np.eye(n_obs)
    
                                                    
    observations = {}
    for k in range(0, analysis_idx + 1, obs_interval):
        if k < len(u_truth):
            y_true = u_truth[k, obs_indices]
            noise = obs_noise_std * np.random.normal(0, 1, n_obs)
            observations[k] = y_true + noise
    
                     
    enkf = EnKF(system, N_ensemble, Q_std=5e-4, R_std=obs_noise_std, Q_d=Q_d)
    
                          
    x0_est = np.zeros(M)
    xi_ensemble = enkf.initialize_enkf(x0_est, P0_std=0.1)
    
                                
    x_mean_history = np.zeros((analysis_idx + 1, M))
    std_history = np.zeros((analysis_idx + 1, M))
    mse_history = np.zeros(analysis_idx + 1)
    residual_mean_history = np.zeros(analysis_idx + 1)
    spread_history = np.zeros(analysis_idx + 1)
    
    x_mean, std = enkf.get_mean_and_spread(xi_ensemble)
    x_mean_history[0] = x_mean
    std_history[0] = std
    mse_history[0] = np.mean((x_mean - u_truth[0])**2)
    residual_mean_history[0] = np.mean(u_truth[0] - x_mean)
    spread_history[0] = np.mean(std)
    
    if 0 in observations:
        z_0 = observations[0]
        xi_ensemble, _ = enkf.measurement_update(xi_ensemble, z_0, H, R)
        x_mean, std = enkf.get_mean_and_spread(xi_ensemble)
        x_mean_history[0] = x_mean
        std_history[0] = std
        mse_history[0] = np.mean((x_mean - u_truth[0])**2)
        residual_mean_history[0] = np.mean(u_truth[0] - x_mean)
        spread_history[0] = np.mean(std)
    
    da_save_dir = os.path.join(r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_newresult", 
                              f"DA_phase_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}s")
    
                                 
    analysis_data_exists = False
    if os.path.exists(da_save_dir):
        try:
            required_files = ["ensemble_mean.csv", "ensemble_std.csv", "metrics.csv", "final_ensemble.csv"]
            if all(os.path.exists(os.path.join(da_save_dir, f)) for f in required_files):
                print("Found existing analysis data, loading...")
                
                                       
                x_mean_history = pd.read_csv(os.path.join(da_save_dir, "ensemble_mean.csv"), header=None).values
                std_history = pd.read_csv(os.path.join(da_save_dir, "ensemble_std.csv"), header=None).values
                metrics_df = pd.read_csv(os.path.join(da_save_dir, "metrics.csv"))
                final_ensemble = pd.read_csv(os.path.join(da_save_dir, "final_ensemble.csv"), header=None).values
                
                                 
                mse_history = metrics_df['mse'].values
                residual_mean_history = metrics_df['residual_mean'].values
                spread_history = metrics_df['spread'].values
                
                                          
                xi_ensemble = final_ensemble 
                
                if (x_mean_history.shape[0] == analysis_idx + 1 and 
                    x_mean_history.shape[1] == M and
                    xi_ensemble.shape[0] == M and
                    xi_ensemble.shape[1] == N_ensemble):
                    
                    analysis_data_exists = True
                    da_time = 0.0                                          
                    print(f"Successfully loaded analysis data from: {da_save_dir}")
                    print(f"Analysis data shape: mean={x_mean_history.shape}, ensemble={xi_ensemble.shape}")
                else:
                    print("Analysis data dimensions don't match current parameters, will regenerate")
                    
        except Exception as e:
            print(f"Error loading analysis data: {e}")
            print("Will regenerate analysis data")
    
                                                     
    if not analysis_data_exists:
        print("Phase 1: Data Assimilation - Computing new analysis")
        start_time = time.time()
        for k in range(analysis_idx):
            if (k+1) % 1000 == 0:
                print(f"  DA Progress: {(k+1)/analysis_idx*100:.0f}%, MSE = {mse_history[k]:.6f}")
            
                         
            xi_ensemble = enkf.time_update(xi_ensemble)
            
                                
            if (k+1) in observations:
                z_k = observations[k+1]
                xi_ensemble, K_k = enkf.measurement_update(xi_ensemble, z_k, H, R)
            
            x_mean, std = enkf.get_mean_and_spread(xi_ensemble)
            x_mean_history[k+1] = x_mean
            std_history[k+1] = std
            mse_history[k+1] = np.mean((x_mean - u_truth[k+1])**2)
            residual_mean_history[k+1] = np.mean(u_truth[k+1] - x_mean)
            spread_history[k+1] = np.mean(std)
        
        da_time = time.time() - start_time
        print(f"  Data Assimilation completed in {da_time:.2f}s")
        
                                        
        if save_data:
            os.makedirs(da_save_dir, exist_ok=True)
            
                                               
            pd.DataFrame(x_mean_history).to_csv(
                os.path.join(da_save_dir, "ensemble_mean.csv"), 
                index=False, header=False, float_format='%.8e')
            
            pd.DataFrame(std_history).to_csv(
                os.path.join(da_save_dir, "ensemble_std.csv"), 
                index=False, header=False, float_format='%.8e')
            
                          
            metrics_df = pd.DataFrame({
                'time': np.arange(analysis_idx + 1) * dt,
        'mse': mse_history,
        'residual_mean': residual_mean_history,
                'spread': spread_history
            })
            metrics_df.to_csv(os.path.join(da_save_dir, "metrics.csv"), index=False)
            
                                                  
            pd.DataFrame(xi_ensemble).to_csv(
                os.path.join(da_save_dir, "final_ensemble.csv"), 
                index=False, header=False, float_format='%.8e')
            
            print(f"Data assimilation results saved to: {da_save_dir}")
    else:
        print("Phase 1: Data Assimilation - Skipped (loaded from existing data)")
    
                                                           
    print("Phase 2: Pure Prediction")
    
                                            
    if selected_members is not None:
        if isinstance(selected_members, int):
            selected_members = [selected_members]
        selected_ensemble = xi_ensemble[:, selected_members]
        print(f"Using selected ensemble members: {selected_members}")
    else:
        if n_prediction_ensemble is not None and n_prediction_ensemble < N_ensemble:
            selected_indices = np.random.choice(N_ensemble, n_prediction_ensemble, replace=False)
            selected_ensemble = xi_ensemble[:, selected_indices]
            print(f"Using {n_prediction_ensemble} randomly selected ensemble members")
        else:
            selected_ensemble = xi_ensemble
            print(f"Using all {N_ensemble} ensemble members")
    
                                              
    if alpha != 1.0:
        x_T = u_truth[analysis_idx]  
        for i in range(selected_ensemble.shape[1]):
            selected_ensemble[:, i] = adjust_ensemble_member(x_T, selected_ensemble[:, i], alpha)
        print(f"Applied alpha adjustment with α = {alpha}")
    
                      
    prediction_steps = lead_idx - analysis_idx
    prediction_mean_history = np.zeros((prediction_steps + 1, M))
    prediction_std_history = np.zeros((prediction_steps + 1, M))
    prediction_mse_history = np.zeros(prediction_steps + 1)
    
                                  
    pred_mean, pred_std = np.mean(selected_ensemble, axis=1), np.std(selected_ensemble, axis=1)
    prediction_mean_history[0] = pred_mean
    prediction_std_history[0] = pred_std
    prediction_mse_history[0] = np.mean((pred_mean - u_truth[analysis_idx])**2)
    
                                               
    start_time = time.time()
    current_ensemble = selected_ensemble.copy()
    
                                                                                                                  
    if abs(alpha) >= 1e-12:
        enkf_pred = EnKF(system, current_ensemble.shape[1], Q_std=enkf.Q_std, R_std=obs_noise_std, Q_d=Q_d)
    
    for k in range(prediction_steps):
        if (k+1) % 1000 == 0:
            print(f"  Prediction Progress: {(k+1)/prediction_steps*100:.0f}%, MSE = {prediction_mse_history[k]:.6f}")
        
                                                 
        if abs(alpha) < 1e-12:
            current_ensemble = system.M_op(current_ensemble)
        else:
            current_ensemble = enkf_pred.time_update(current_ensemble)
        
        pred_mean, pred_std = np.mean(current_ensemble, axis=1), np.std(current_ensemble, axis=1)
        prediction_mean_history[k+1] = pred_mean
        prediction_std_history[k+1] = pred_std
        
        truth_idx = analysis_idx + k + 1
        if truth_idx < len(u_truth):
            prediction_mse_history[k+1] = np.mean((pred_mean - u_truth[truth_idx])**2)
    
    prediction_time = time.time() - start_time
    print(f"  Pure prediction completed in {prediction_time:.2f}s")
    
                     
    total_length = analysis_idx + 1 + prediction_steps
    
                         
    total_time_history = np.zeros(total_length)
    total_time_history[:analysis_idx + 1] = np.arange(analysis_idx + 1) * dt
    total_time_history[analysis_idx + 1:] = analysis_time + np.arange(1, prediction_steps + 1) * dt
    
    total_mean_history = np.zeros((total_length, M))
    total_mean_history[:analysis_idx + 1] = x_mean_history
    total_mean_history[analysis_idx + 1:] = prediction_mean_history[1:]
    
    total_std_history = np.zeros((total_length, M))
    total_std_history[:analysis_idx + 1] = std_history
    total_std_history[analysis_idx + 1:] = prediction_std_history[1:]
    
    total_mse_history = np.zeros(total_length)
    total_mse_history[:analysis_idx + 1] = mse_history
    total_mse_history[analysis_idx + 1:] = prediction_mse_history[1:]
    
    return {
        'u_truth': u_truth,
        'x_mean_total': total_mean_history,
        'std_total': total_std_history,
        'mse_total': total_mse_history,
        'time_total': total_time_history,
        'x_mean_da': x_mean_history,
        'std_da': std_history,
        'mse_da': mse_history,
        'x_mean_pred': prediction_mean_history,
        'std_pred': prediction_std_history,
        'mse_pred': prediction_mse_history,
        'x': x,
        'analysis_idx': analysis_idx,
        'lead_idx': lead_idx,
        'target_idx': target_idx,
        'ensemble_final_da': xi_ensemble,
        'ensemble_final_pred': current_ensemble,
        'selected_members': selected_members,
        'alpha': alpha,
        'da_time': da_time,
        'prediction_time': prediction_time,
        'da_save_dir': da_save_dir if save_data else None
    }


def create_enhanced_figures(target_time, analysis_time, lead_time, N_ensemble, Q_d, 
                          selected_members=None, alpha=1.0, n_prediction_ensemble=None,
                          save_data=True, time_chosen=None):
    
    save_dir = os.path.join(r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_newresult", 
                           f"Enhanced_results_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}s_lead{lead_time:.1f}s_alpha{alpha}_Npred{n_prediction_ensemble}")
    os.makedirs(save_dir, exist_ok=True)
    
    results = run_enkf_enhanced_experiment(
        target_time, analysis_time, lead_time, N_ensemble, Q_d,
        selected_members, alpha, n_prediction_ensemble, save_data
    )
    
    u_truth = results['u_truth']
    x_mean_total = results['x_mean_total']
    std_total = results['std_total']
    mse_total = results['mse_total']
    time_total = results['time_total']
    x = results['x']
    analysis_idx = results['analysis_idx']
    lead_idx = results['lead_idx']
    
    dt = 1e-4
    
                            
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    selected_idx = int(round(selected_time_s / dt))
    selected_idx = max(0, min(selected_idx, len(u_truth) - 1))
    
                                               
    if selected_time_s <= analysis_time:
        plot_idx = int(selected_time_s / dt)
    else:
        plot_idx = analysis_idx + int((selected_time_s - analysis_time) / dt)
    plot_idx = max(0, min(plot_idx, len(x_mean_total) - 1))
    
    selected_truth = u_truth[selected_idx]
    selected_est = x_mean_total[plot_idx]
    selected_std = std_total[plot_idx]
    
                                            
    plt.figure(figsize=(12, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=2.5, label='Truth')
    plt.plot(x, selected_est, 'r--', linewidth=2.5, label='Enhanced EnKF Prediction')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF Prediction vs Truth at t = {selected_time_s:.2f}s(Analysis until t={analysis_time:.1f}s, α={alpha}, N_pred={n_prediction_ensemble})', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_vs_truth_at_t_{selected_time_s:.2f}s_N{N_ensemble}_alpha{alpha}_Npred{n_prediction_ensemble}.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                     
    plt.figure(figsize=(12, 6))
    plt.semilogy(time_total, mse_total, 'g-', linewidth=2.5, label='MSE')
    plt.axvline(x=analysis_time, color='orange', linestyle='--', linewidth=2, 
                label=f'Analysis end (t={analysis_time:.2f}s)')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, 
                label=f'Evaluation time (t={selected_time_s:.2f}s)')
    plt.xlabel('Time t (s)', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title(f'MSE Evolution: Data Assimilation + Pure Prediction (N_pred={n_prediction_ensemble})', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mse_evolution_enhanced_N{N_ensemble}_alpha{alpha}_Npred{n_prediction_ensemble}.png'), dpi=300, bbox_inches='tight')
                            
    
                                               
    plt.figure(figsize=(14, 8))
    plt.plot(x, selected_truth, 'b-', linewidth=3, label='Truth', zorder=5)
    plt.plot(x, selected_est, 'r--', linewidth=2.5, label='Enhanced EnKF', zorder=4)
    
                  
    lower_2sigma = selected_est - 2*selected_std
    upper_2sigma = selected_est + 2*selected_std
    
    plt.fill_between(x, lower_2sigma, upper_2sigma, 
                     alpha=0.4, color='red', zorder=2, label='±2σ uncertainty')
    
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'EnKF with Error Bands at t = {selected_time_s:.2f}s ({phase_info})' + 
              f'Analysis until t={analysis_time:.1f}s, α={alpha}, N_pred={n_prediction_ensemble}', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_with_error_bands_N{N_ensemble}_alpha{alpha}_Npred{n_prediction_ensemble}.png'), dpi=300, bbox_inches='tight')
                            
    
                                                                              
    plt.figure(figsize=(14, 8))
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=5)
    
                                                      
                                                                                                      
    if selected_time_s <= analysis_time:
                                                                                  
        ensemble_snapshot = results['ensemble_final_da']
        phase_info = 'Data Assimilation'
    else:
        ensemble_snapshot = results['ensemble_final_pred']
        phase_info = 'Pure Prediction'
    
    num_members = ensemble_snapshot.shape[1]
    member_colors = plt.cm.viridis(np.linspace(0, 1, num_members))
    for j in range(num_members):
        plt.plot(x, ensemble_snapshot[:, j], color=member_colors[j], linewidth=1.2, alpha=0.6, label = f'Member {j+1}')
    
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'All Ensemble Members vs Truth at t = {selected_time_s:.2f}s ({phase_info})' +
              f' (α={alpha}, N_pred={n_prediction_ensemble})', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
                                                
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ensemble_members_vs_truth_t_{selected_time_s:.2f}s_N{N_ensemble}_alpha{alpha}_Npred{n_prediction_ensemble}.png'),
                dpi=300, bbox_inches='tight')
                            

                                                                                
    plt.figure(figsize=(14, 8))
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
                                                 
    if selected_time_s <= analysis_time:
        envelope_snapshot = results['ensemble_final_da']
        phase_info_env = 'Data Assimilation'
    else:
        envelope_snapshot = results['ensemble_final_pred']
        phase_info_env = 'Pure Prediction'
                                          
    lower_envelope = np.min(envelope_snapshot, axis=1)
    upper_envelope = np.max(envelope_snapshot, axis=1)
                                      
    plt.plot(x, selected_est, 'r--', linewidth=2.0, label='EnKF')
    plt.fill_between(x, lower_envelope, upper_envelope, color='tab:blue', alpha=0.2, label='Min–Max envelope')
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF with Min–Max Error Bands at t = {selected_time_s:.2f}s ({phase_info_env})' +
              f' (Analysis until t={analysis_time:.1f}s, α={alpha}, N_pred={n_prediction_ensemble})',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_with_minmax_envelope_N{N_ensemble}_alpha{alpha}_Npred{n_prediction_ensemble}.png'),
                dpi=300, bbox_inches='tight')
                            

                                   
    rmse_at_time = compute_rmse_over_domain(selected_truth, selected_est)
    rogue_threshold = 1.0
    rogue_region = selected_truth > rogue_threshold
    if not np.any(rogue_region):
        rogue_region = None
    peak_amp_error = compute_peak_amplitude_error(selected_truth, selected_est, rogue_region)
    mean_abs_err_peak_region = compute_mean_abs_error_region(selected_truth, selected_est, rogue_region)

    print(f"n=== Enhanced EnKF Results ===")
    print(f"Analysis phase: 0 to {analysis_time:.2f}s")
    print(f"Prediction phase: {analysis_time:.2f}s to {lead_time:.2f}s")
    print(f"Alpha adjustment: {alpha}")
    print(f"Selected members: {selected_members}")
    print(f"RMSE at t={selected_time_s:.2f}s: {rmse_at_time:.6f}")
    if rogue_region is not None:
        print(f"Mean absolute error in rogue region: {mean_abs_err_peak_region:.6f}")
    else:
        print("No rogue region detected")
    
                                
    results_summary = {
        'analysis_time': analysis_time,
        'lead_time': lead_time,
        'target_time': target_time,
        'evaluation_time': selected_time_s,
        'N_ensemble': N_ensemble,
        'Q_d': Q_d,
        'alpha': alpha,
        'selected_members': str(selected_members),
        'n_prediction_ensemble': n_prediction_ensemble,
        'rmse_at_evaluation': rmse_at_time,
        'peak_amplitude_error': peak_amp_error,
        'mean_abs_error_peak_region': mean_abs_err_peak_region,
        'da_time': results['da_time'],
        'prediction_time': results['prediction_time'],
        'final_mse_da': results['mse_da'][-1],
        'final_mse_pred': results['mse_pred'][-1] if len(results['mse_pred']) > 1 else np.nan
    }
    
                           
    time_series_df = pd.DataFrame({
        'time': time_total,
        'mse': mse_total,
        'mean_amplitude': np.mean(x_mean_total, axis=1),
        'mean_std': np.mean(std_total, axis=1)
    })
    time_series_df.to_csv(os.path.join(save_dir, 'time_series_data.csv'), index=False)
    
                  
    pd.DataFrame([results_summary]).to_csv(os.path.join(save_dir, 'experiment_summary.csv'), index=False)
    
    print(f"All results saved to: {save_dir}")
    print(f"Data assimilation results saved to: {results['da_save_dir']}")
    
    return results

def compare_alpha_values(target_time, analysis_time, lead_time, N_ensemble, Q_d,
                        alpha_values, selected_members=None, n_prediction_ensemble=None,
                        save_data=True, time_chosen=None):
    
    results_dict = {}
    
    print(f"\n=== Comparing Alpha Values: {alpha_values} ===")
    
    for alpha in alpha_values:
                                                             
        np.random.seed(40)
        random.seed(40)
        
        print(f"\n--- Running experiment with α = {alpha} ---")
        
        results = run_enkf_enhanced_experiment(
            target_time, analysis_time, lead_time, N_ensemble, Q_d,
            selected_members, alpha, n_prediction_ensemble, save_data
        )
        
        results_dict[alpha] = results
    
    compare_dir = os.path.join(
        r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_newresult",
        f"Alpha_comparison_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}s_lead{lead_time:.1f}s_Npred{n_prediction_ensemble}"
    )
    os.makedirs(compare_dir, exist_ok=True)
    
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    
                        
    ref_results = results_dict[alpha_values[0]]
    x = ref_results['x']
    u_truth = ref_results['u_truth']
    dt = 1e-4
    
    selected_idx = int(round(selected_time_s / dt))
    selected_idx = max(0, min(selected_idx, len(u_truth) - 1))
    selected_truth = u_truth[selected_idx]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))
    
                                 
    plt.figure(figsize=(14, 8))
    
    for i, alpha in enumerate(alpha_values):
        results = results_dict[alpha]
        time_total = results['time_total']
        mse_total = results['mse_total']
        
        plt.semilogy(time_total, mse_total, color=colors[i], linewidth=2.5, 
                    label=f'α = {alpha}', alpha=0.8)
    
    plt.axvline(x=analysis_time, color='red', linestyle='--', linewidth=2, 
                label=f'Analysis end (t={analysis_time:.2f}s)', alpha=0.7)
    plt.axvline(x=selected_time_s, color='orange', linestyle='--', linewidth=2, 
                label=f'Evaluation time (t={selected_time_s:.2f}s)', alpha=0.7)
    
    plt.xlabel('Time t (s)', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title(f'MSE Evolution Comparison: Different α Values(N_pred={n_prediction_ensemble}, Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'mse_evolution_comparison_alphas.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                        
    plt.figure(figsize=(14, 8))
    
                
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
    
                                     
    for i, alpha in enumerate(alpha_values):
        results = results_dict[alpha]
        
                                                   
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        plt.plot(x, selected_est, color=colors[i], linestyle='--', linewidth=2.5, 
                label=f'α = {alpha}', alpha=0.8)
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'Predictions vs Truth Comparison at t = {selected_time_s:.2f}s ({phase_info})(N_pred={n_prediction_ensemble}, Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'predictions_vs_truth_comparison_alphas_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                                
    plt.figure(figsize=(14, 8))
    
                
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
    
                                                      
    for i, alpha in enumerate(alpha_values):
        results = results_dict[alpha]
        
                                                   
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        selected_std = results['std_total'][plot_idx]
        
                         
        plt.plot(x, selected_est, color=colors[i], linestyle='--', linewidth=2.5, 
                label=f'α = {alpha}', alpha=0.8)
        
                                
    lower_2sigma = selected_est - 2*selected_std
    upper_2sigma = selected_est + 2*selected_std
    
    plt.fill_between(x, lower_2sigma, upper_2sigma, 
                        color=colors[i], alpha=0.2, zorder=1)
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'Predictions with Error Bands Comparison at t = {selected_time_s:.2f}s ({phase_info})(N_pred={n_prediction_ensemble}, Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'predictions_with_error_bands_comparison_alphas_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                                                        
    summary_data = []
    rmse_values = []
    
    for alpha in alpha_values:
        results = results_dict[alpha]
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        rmse = compute_rmse_over_domain(selected_truth, selected_est)
        rmse_values.append(rmse)
        
                                                         
        time_series_df = pd.DataFrame({
            'time': results['time_total'],
            'mse': results['mse_total'],
            'mean_amplitude': np.mean(results['x_mean_total'], axis=1),
            'mean_std': np.mean(results['std_total'], axis=1)
        })
        time_series_df.to_csv(os.path.join(compare_dir, f'time_series_alpha_{alpha}.csv'), index=False)
        
                                            
        prediction_df = pd.DataFrame({
            'x': x,
            'truth': selected_truth,
            'prediction': selected_est,
            'std': results['std_total'][plot_idx]
        })
        prediction_df.to_csv(os.path.join(compare_dir, f'prediction_alpha_{alpha}_t_{selected_time_s:.2f}s.csv'), index=False)
        
        summary_data.append({
            'alpha': alpha,
            'rmse_at_evaluation': rmse,
            'final_mse_da': results['mse_da'][-1],
            'final_mse_pred': results['mse_pred'][-1] if len(results['mse_pred']) > 1 else np.nan,
            'da_time': results['da_time'],
            'prediction_time': results['prediction_time'],
            'time_series_file': f'time_series_alpha_{alpha}.csv',
            'prediction_file': f'prediction_alpha_{alpha}_t_{selected_time_s:.2f}s.csv'
        })
    
                             
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(compare_dir, 'alpha_comparison_summary.csv'), index=False)
    
    print(f"=== Alpha Comparison Results ===")
    print(f"Results saved to: {compare_dir}")
    print("RMSE Summary:")
    for alpha, rmse in zip(alpha_values, rmse_values):
        print(f"  α = {alpha:4.1f}: RMSE = {rmse:.6f}")
    print(f"Generated files:")
    print(f"  - 3 comparison plots")
    print(f"  - {len(alpha_values)} time series CSV files")
    print(f"  - {len(alpha_values)} prediction CSV files")
    print(f"  - 1 summary CSV file")
    
    return results_dict, compare_dir

def compare_alpha_npred_combinations(target_time, analysis_time, lead_time, N_ensemble, Q_d,
                                   alpha_values, n_prediction_values, selected_members=None,
                                   save_data=True, time_chosen=None):
    
    results_dict = {}
    
    print(f"\n=== Comparing Alpha and N_pred Combinations ===")
    print(f"Alpha values: {alpha_values}")
    print(f"N_pred values: {n_prediction_values}")

    for alpha in alpha_values:
        for n_pred in n_prediction_values:
                                                                       
            np.random.seed(40)
            random.seed(40)
            
            combo_key = f"α{alpha}_N{n_pred}"
            print(f"\n--- Running experiment with α = {alpha}, N_pred = {n_pred} ---")
            
            results = run_enkf_enhanced_experiment(
                target_time, analysis_time, lead_time, N_ensemble, Q_d,
                selected_members, alpha, n_pred, save_data
            )
            
            results_dict[combo_key] = {
                'results': results,
                'alpha': alpha,
                'n_pred': n_pred
            }

            print(f"--- Generating individual figures for α = {alpha}, N_pred = {n_pred} ---")
            individual_results = create_enhanced_figures(
                target_time, analysis_time, lead_time, N_ensemble, Q_d,
                selected_members, alpha, n_pred, save_data=False, time_chosen=time_chosen
            )
    
    compare_dir = os.path.join(
        r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_newresult",
        f"Alpha_Npred_comparison_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}s_lead{lead_time:.1f}s"
    )
    os.makedirs(compare_dir, exist_ok=True)
    
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    
    first_key = list(results_dict.keys())[0]
    ref_results = results_dict[first_key]['results']
    x = ref_results['x']
    u_truth = ref_results['u_truth']
    dt = 1e-4
    
    selected_idx = int(round(selected_time_s / dt))
    selected_idx = max(0, min(selected_idx, len(u_truth) - 1))
    selected_truth = u_truth[selected_idx]
    
    n_combos = len(results_dict)
    colors = plt.cm.tab10(np.linspace(0, 1, n_combos))
    
                                 
    plt.figure(figsize=(16, 10))
    
    for i, (combo_key, combo_data) in enumerate(results_dict.items()):
        results = combo_data['results']
        alpha = combo_data['alpha']
        n_pred = combo_data['n_pred']
        
        time_total = results['time_total']
        mse_total = results['mse_total']
        
        plt.semilogy(time_total, mse_total, color=colors[i], 
                    linewidth=2.5, label=f'α={alpha}, N_pred={n_pred}', alpha=0.8)
    
    plt.axvline(x=analysis_time, color='red', linestyle='--', linewidth=2, 
                label=f'Analysis end (t={analysis_time:.2f}s)', alpha=0.7)
    plt.axvline(x=selected_time_s, color='orange', linestyle='--', linewidth=2, 
                label=f'Evaluation time (t={selected_time_s:.2f}s)', alpha=0.7)
    
    plt.xlabel('Time t (s)', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title(f'MSE Evolution: α and N_pred Combinations(Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'mse_evolution_comparison_combinations.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                        
    plt.figure(figsize=(16, 10))
    
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
    
    for i, (combo_key, combo_data) in enumerate(results_dict.items()):
        results = combo_data['results']
        alpha = combo_data['alpha']
        n_pred = combo_data['n_pred']
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        plt.plot(x, selected_est, color=colors[i], 
                linewidth=2.5, label=f'α={alpha}, N_pred={n_pred}', alpha=0.8)
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'Predictions vs Truth at t = {selected_time_s:.2f}s ({phase_info})(Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'predictions_vs_truth_comparison_combinations_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                                
    plt.figure(figsize=(16, 10))
    
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
    
    for i, (combo_key, combo_data) in enumerate(results_dict.items()):
        results = combo_data['results']
        alpha = combo_data['alpha']
        n_pred = combo_data['n_pred']
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        selected_std = results['std_total'][plot_idx]
        
        plt.plot(x, selected_est, color=colors[i], 
                linewidth=2.5, label=f'α={alpha}, N_pred={n_pred}', alpha=0.8)
        
        lower_2sigma = selected_est - 2*selected_std
        upper_2sigma = selected_est + 2*selected_std
        
        plt.fill_between(x, lower_2sigma, upper_2sigma, 
                        color=colors[i], alpha=0.15, zorder=1)
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'Predictions with Error Bands at t = {selected_time_s:.2f}s ({phase_info})(Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'predictions_with_error_bands_comparison_combinations_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    

    plt.figure(figsize=(12, 8))
    
    rmse_matrix = np.zeros((len(alpha_values), len(n_prediction_values)))
    
    for i, alpha in enumerate(alpha_values):
        for j, n_pred in enumerate(n_prediction_values):
            combo_key = f"α{alpha}_N{n_pred}"
            results = results_dict[combo_key]['results']
            
            if selected_time_s <= analysis_time:
                plot_idx = int(selected_time_s / dt)
            else:
                plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
            plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
            
            selected_est = results['x_mean_total'][plot_idx]
            rmse = compute_rmse_over_domain(selected_truth, selected_est)
            rmse_matrix[i, j] = rmse
    
    im = plt.imshow(rmse_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='RMSE')
    
    plt.xlabel('N_prediction_ensemble', fontsize=14)
    plt.ylabel('Alpha (α)', fontsize=14)
    plt.title(f'RMSE Heatmap at t = {selected_time_s:.2f}s', fontsize=16, fontweight='bold')
    
    plt.xticks(range(len(n_prediction_values)), n_prediction_values)
    plt.yticks(range(len(alpha_values)), alpha_values)

    for i in range(len(alpha_values)):
        for j in range(len(n_prediction_values)):
            plt.text(j, i, f'{rmse_matrix[i, j]:.4f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'rmse_heatmap_combinations_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
    summary_data = []
    
    for combo_key, combo_data in results_dict.items():
        results = combo_data['results']
        alpha = combo_data['alpha']
        n_pred = combo_data['n_pred']
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        rmse = compute_rmse_over_domain(selected_truth, selected_est)

        time_series_df = pd.DataFrame({
            'time': results['time_total'],
            'mse': results['mse_total'],
            'mean_amplitude': np.mean(results['x_mean_total'], axis=1),
            'mean_std': np.mean(results['std_total'], axis=1)
        })
        time_series_df.to_csv(os.path.join(compare_dir, f'time_series_{combo_key}.csv'), index=False)
        
        prediction_df = pd.DataFrame({
            'x': x,
            'truth': selected_truth,
            'prediction': selected_est,
            'std': results['std_total'][plot_idx]
        })
        prediction_df.to_csv(os.path.join(compare_dir, f'prediction_{combo_key}_t_{selected_time_s:.2f}s.csv'), index=False)
        
        summary_data.append({
            'alpha': alpha,
            'n_prediction_ensemble': n_pred,
            'rmse_at_evaluation': rmse,
            'final_mse_da': results['mse_da'][-1],
            'final_mse_pred': results['mse_pred'][-1] if len(results['mse_pred']) > 1 else np.nan,
            'da_time': results['da_time'],
            'prediction_time': results['prediction_time'],
            'time_series_file': f'time_series_{combo_key}.csv',
            'prediction_file': f'prediction_{combo_key}_t_{selected_time_s:.2f}s.csv'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(compare_dir, 'combinations_comparison_summary.csv'), index=False)
    
    print(f"=== Alpha and N_pred Combinations Results ===")
    print(f"Results saved to: {compare_dir}")
    print("RMSE Summary:")
    for combo_key, combo_data in results_dict.items():
        alpha = combo_data['alpha']
        n_pred = combo_data['n_pred']
        results = combo_data['results']
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        rmse = compute_rmse_over_domain(selected_truth, selected_est)
        print(f"  α={alpha:4.2f}, N_pred={n_pred:3d}: RMSE = {rmse:.6f}")
    
    print(f"Generated files:")
    print(f"  - 4 comparison plots (including heatmap)")
    print(f"  - {len(results_dict)} time series CSV files")
    print(f"  - {len(results_dict)} prediction CSV files")
    print(f"  - 1 summary CSV file")
    
    return results_dict, compare_dir

def compare_n_prediction_values(target_time, analysis_time, lead_time, N_ensemble, Q_d,
                               n_prediction_values, selected_members=None, alpha=1.0,
                               save_data=True, time_chosen=None):
    
    results_dict = {}
    
    print(f"\n=== Comparing N_prediction Values: {n_prediction_values} ===")
    
    for n_pred in n_prediction_values:
                                                              
        np.random.seed(40)
        random.seed(40)
        
        print(f"\n--- Running experiment with N_pred = {n_pred} ---")
        
        results = run_enkf_enhanced_experiment(
            target_time, analysis_time, lead_time, N_ensemble, Q_d,
            selected_members, alpha, n_pred, save_data
        )
        
        results_dict[n_pred] = results
    
    compare_dir = os.path.join(
        r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_newresult",
        f"Npred_comparison_N{N_ensemble}_Qd{Q_d}_analysis{analysis_time:.1f}s_lead{lead_time:.1f}s_alpha{alpha}"
    )
    os.makedirs(compare_dir, exist_ok=True)
    
    selected_time_s = float(time_chosen) if time_chosen is not None else float(lead_time)
    
    ref_results = results_dict[n_prediction_values[0]]
    x = ref_results['x']
    u_truth = ref_results['u_truth']
    dt = 1e-4
    
    selected_idx = int(round(selected_time_s / dt))
    selected_idx = max(0, min(selected_idx, len(u_truth) - 1))
    selected_truth = u_truth[selected_idx]
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(n_prediction_values)))
    
                                 
    plt.figure(figsize=(14, 8))
    
    for i, n_pred in enumerate(n_prediction_values):
        results = results_dict[n_pred]
        time_total = results['time_total']
        mse_total = results['mse_total']
        
        plt.semilogy(time_total, mse_total, color=colors[i], linewidth=2.5, 
                    label=f'N_pred = {n_pred}', alpha=0.8)
    
    plt.axvline(x=analysis_time, color='red', linestyle='--', linewidth=2, 
                label=f'Analysis end (t={analysis_time:.2f}s)', alpha=0.7)
    plt.axvline(x=selected_time_s, color='orange', linestyle='--', linewidth=2, 
                label=f'Evaluation time (t={selected_time_s:.2f}s)', alpha=0.7)
    
    plt.xlabel('Time t (s)', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title(f'MSE Evolution Comparison: Different N_pred Values(α={alpha}, Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'mse_evolution_comparison_nprods.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                                                      
    for i, n_pred in enumerate(n_prediction_values):
        results = results_dict[n_pred]
        if selected_time_s <= analysis_time:
            ensemble_snapshot = results['ensemble_final_da']
            phase_info_single = 'DA snapshot'
        else:
            ensemble_snapshot = results['ensemble_final_pred']
            phase_info_single = 'Prediction snapshot'
        
        plt.figure(figsize=(14, 8))
        plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=5)
        num_members = ensemble_snapshot.shape[1]
        member_colors = plt.cm.viridis(np.linspace(0, 1, num_members))
        for j in range(num_members):
            plt.plot(x, ensemble_snapshot[:, j], color=member_colors[j], linewidth=1.2, alpha=0.6)
        
        plt.xlabel('Spatial Position x', fontsize=14)
        plt.ylabel('Wave Amplitude u', fontsize=14)
        plt.title(f'Ensemble Members vs Truth (N_pred={n_pred}) at t = {selected_time_s:.2f}s ({phase_info_single})' +
                  f' (α={alpha})', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(compare_dir, f'ensemble_members_vs_truth_npred_{n_pred}_t_{selected_time_s:.2f}s.png'),
                    dpi=300, bbox_inches='tight')
                                

                                        
    plt.figure(figsize=(14, 8))
    
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
    
    for i, n_pred in enumerate(n_prediction_values):
        results = results_dict[n_pred]
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        plt.plot(x, selected_est, color=colors[i], linestyle='--', linewidth=2.5, 
                label=f'N_pred = {n_pred}', alpha=0.8)
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'Predictions vs Truth Comparison at t = {selected_time_s:.2f}s ({phase_info})(α={alpha}, Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'predictions_vs_truth_comparison_nprods_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                                
    plt.figure(figsize=(14, 8))
    
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
    
    for i, n_pred in enumerate(n_prediction_values):
        results = results_dict[n_pred]
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        selected_std = results['std_total'][plot_idx]
        
        plt.plot(x, selected_est, color=colors[i], linestyle='--', linewidth=2.5, 
                label=f'N_pred = {n_pred}', alpha=0.8)
        
        lower_2sigma = selected_est - 2*selected_std
        upper_2sigma = selected_est + 2*selected_std
        
        plt.fill_between(x, lower_2sigma, upper_2sigma, 
                        color=colors[i], alpha=0.2, zorder=1)
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'Predictions with Error Bands Comparison at t = {selected_time_s:.2f}s ({phase_info})(α={alpha}, Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'predictions_with_error_bands_comparison_nprods_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                                                   
    plt.figure(figsize=(14, 8))
    
    plt.plot(x, selected_truth, 'k-', linewidth=3, label='Truth', zorder=10)
    
    mean_curves = []
    for n_pred in n_prediction_values:
        results = results_dict[n_pred]
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        mean_curves.append(results['x_mean_total'][plot_idx])
    
    means_stack = np.vstack(mean_curves)                                        
    lower_means = np.min(means_stack, axis=0)
    upper_means = np.max(means_stack, axis=0)
    
    plt.fill_between(x, lower_means, upper_means, color='tab:blue', alpha=0.2,
                     label='Min–Max of ensemble means')
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    phase_info = "Data Assimilation" if selected_time_s <= analysis_time else "Pure Prediction"
    plt.title(f'Predictions with Error Bands Comparison at t = {selected_time_s:.2f}s ({phase_info})(α={alpha}, Analysis until t={analysis_time:.1f}s)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, f'mean_envelope_across_nprods_t_{selected_time_s:.2f}s.png'), 
                dpi=300, bbox_inches='tight')
                            
    
                                                         
    summary_data = []
    rmse_values = []
    
    for n_pred in n_prediction_values:
        results = results_dict[n_pred]
        
        if selected_time_s <= analysis_time:
            plot_idx = int(selected_time_s / dt)
        else:
            plot_idx = results['analysis_idx'] + int((selected_time_s - analysis_time) / dt)
        plot_idx = max(0, min(plot_idx, len(results['x_mean_total']) - 1))
        
        selected_est = results['x_mean_total'][plot_idx]
        rmse = compute_rmse_over_domain(selected_truth, selected_est)
        rmse_values.append(rmse)
        
        time_series_df = pd.DataFrame({
            'time': results['time_total'],
            'mse': results['mse_total'],
            'mean_amplitude': np.mean(results['x_mean_total'], axis=1),
            'mean_std': np.mean(results['std_total'], axis=1)
        })
        time_series_df.to_csv(os.path.join(compare_dir, f'time_series_npred_{n_pred}.csv'), index=False)
        
        prediction_df = pd.DataFrame({
            'x': x,
            'truth': selected_truth,
            'prediction': selected_est,
            'std': results['std_total'][plot_idx]
        })
        prediction_df.to_csv(os.path.join(compare_dir, f'prediction_npred_{n_pred}_t_{selected_time_s:.2f}s.csv'), index=False)
        
        summary_data.append({
            'n_prediction_ensemble': n_pred,
            'rmse_at_evaluation': rmse,
            'final_mse_da': results['mse_da'][-1],
            'final_mse_pred': results['mse_pred'][-1] if len(results['mse_pred']) > 1 else np.nan,
            'da_time': results['da_time'],
            'prediction_time': results['prediction_time'],
            'time_series_file': f'time_series_npred_{n_pred}.csv',
            'prediction_file': f'prediction_npred_{n_pred}_t_{selected_time_s:.2f}s.csv'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(compare_dir, 'npred_comparison_summary.csv'), index=False)
    
    print(f"=== N_prediction Comparison Results ===")
    print(f"Results saved to: {compare_dir}")
    print("RMSE Summary:")
    for n_pred, rmse in zip(n_prediction_values, rmse_values):
        print(f"  N_pred = {n_pred:3d}: RMSE = {rmse:.6f}")
    print(f"Generated files:")
    print(f"  - 4 comparison plots")
    print(f"  - {len(n_prediction_values)} time series CSV files")
    print(f"  - {len(n_prediction_values)} prediction CSV files")
    print(f"  - 1 summary CSV file")
    
    return results_dict, compare_dir

if __name__ == "__main__":
                                        
                                      
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
                                     
       
    
                  
                                                  
                                                       
                                                                         
                                                             
                                                            
                                                     
       
    
                          
                                              
                                                                          
                                                                         
                                                                          
                                                                                       
       

    alpha_values = [1.0]
    n_prediction_values = [150]
    
    results_dict_combo, compare_dir_combo = compare_alpha_npred_combinations(
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