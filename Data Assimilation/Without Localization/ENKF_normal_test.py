import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.linalg import sqrtm

np.random.seed(40)

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

def run_enkf_experiment(target_time, N_ensemble, Q_d):

    L = 50.0
    dx = 0.1
    M = int(L / dx)
    dt = 1e-4
    T = target_time + 0.1
    N_t = int(T / dt)
    x = np.linspace(-L/2, L/2, M, endpoint=False)
    target_idx = int(target_time / dt)
    

    print(f"\n--- Running EnKF with N = {N_ensemble} ---")
    print(f"Target time: {target_time:.2f}s")
    print(f"System: M={M}, dx={dx:.3f}, dt={dt}")
    
    system = mKdVSystem(M, dx, dt)
    
                                     
    truth_filename = f"u_truth_T{T:.1f}_dt{dt}_M{M}_dx{dx:.3f}.csv"
    truth_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_results\truth_data"
    os.makedirs(truth_dir, exist_ok=True)
    truth_path = os.path.join(truth_dir, truth_filename)
    
    if os.path.exists(truth_path):
        print(f"Loading existing truth solution: {truth_filename}")
        import pandas as pd
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
        
                                            
        import pandas as pd
        pd.DataFrame(u_truth).to_csv(truth_path, index=False, header=False, float_format='%.8e')
        print(f"Truth solution saved to: {truth_path}")
    
                  
    obs_ratio = 0.05
    obs_interval = 3000
    obs_noise_std = 0.1
    
    n_obs = int(obs_ratio * M)
    obs_indices = (np.arange(n_obs) * M // n_obs).astype(int)
    
    H = np.zeros((n_obs, M))
    for i, idx in enumerate(obs_indices):
        H[i, idx] = 1.0
    R = (obs_noise_std**2) * np.eye(n_obs)
    
                           
    observations = {}
    for k in range(0, N_t + 1, obs_interval):
        y_true = u_truth[k, obs_indices]
        noise = obs_noise_std * np.random.normal(0, 1, n_obs)
        observations[k] = y_true + noise
    
                     
    enkf = EnKF(system, N_ensemble, Q_std=5e-4, R_std=obs_noise_std, Q_d=Q_d)
    
                          
    x0_est = np.zeros(M)
    xi_ensemble = enkf.initialize_enkf(x0_est, P0_std=0.1)
    
    x_mean_history = np.zeros((N_t + 1, M))
    std_history = np.zeros((N_t + 1, M))
    mse_history = np.zeros(N_t + 1)
    residual_mean_history = np.zeros(N_t + 1)
    spread_history = np.zeros(N_t + 1)
    
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
    
    start_time = time.time()
    for k in range(N_t):
        if (k+1) % 1000 == 0:
            print(f"  Progress: {(k+1)/N_t*100:.0f}%, MSE = {mse_history[k]:.6f}")
        
                     
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
    
    running_time = time.time() - start_time
    print(f"  Completed in {running_time:.2f}s")
    
    return {
        'u_truth': u_truth,
        'x_mean': x_mean_history,
        'std': std_history,
        'mse': mse_history,
        'residual_mean': residual_mean_history,
        'spread': spread_history,
        'x': x,
        'target_idx': target_idx,
        'ensemble_final': xi_ensemble,
        'running_time': running_time
    }

def create_all_figures(target_time, N_ensemble, Q_d, time_chosen=None):

    save_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation\ENKF_normal_results\Large_interval"
    os.makedirs(save_dir, exist_ok=True)
    
    results = run_enkf_experiment(target_time, N_ensemble, Q_d)
    
    u_truth = results['u_truth']
    x_mean = results['x_mean']
    std = results['std']
    mse = results['mse']
    residual_mean = results['residual_mean']
    spread = results['spread']
    x = results['x']
    target_idx = results['target_idx']
    
    dt = 1e-4
    time_vec = np.arange(len(mse)) * dt
                                                                                
    selected_time_s = float(time_chosen) if time_chosen is not None else float(target_time)
    selected_idx = int(round(selected_time_s / dt))
    selected_idx = max(0, min(selected_idx, len(u_truth) - 1))
                                      
    selected_truth = u_truth[selected_idx]
    selected_est = x_mean[selected_idx]
    selected_std = std[selected_idx]
                                                                  
    plot_step = 10
    mask_upto = time_vec <= (selected_time_s + 1e-12)
    t_plot = time_vec[mask_upto][::plot_step]
    mse_plot = mse[mask_upto][::plot_step]
    residual_plot = residual_mean[mask_upto][::plot_step]
    spread_plot = spread[mask_upto][::plot_step]
    
                                    
    plt.figure(figsize=(12, 6))
    plt.plot(x, selected_truth, 'b-', linewidth=2.5, label='Truth')
    plt.plot(x, selected_est, 'r--', linewidth=2.5, label='EnKF Prediction')
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF prediction vs truth (assimilation up to t = {target_time:.2f})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
                                     
                                                         
                                                                      
                                                                         
                           
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_vs_truth_at_t_{selected_time_s:.2f}s_ensemble_{N_ensemble}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    plot_indices = slice(None, None, 10)  
    
                                        
    plt.figure(figsize=(12, 6))
    plt.semilogy(t_plot, mse_plot, 'g-', linewidth=2.5, label='MSE')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, 
                label=f'Target time (t={selected_time_s:.2f}s)')
    plt.xlabel('Time t (s)', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title(f'MSE evolution (assimilation up to t = {target_time:.2f})', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mse_evolution_t_{selected_time_s:.2f}s_ensemble_{N_ensemble}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
                                   
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, residual_plot, 'b-', linewidth=2.5, label='Mean Residual')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, 
                label=f'Target time (t={selected_time_s:.2f}s)')
    plt.xlabel('Time t (s)', fontsize=14)
    plt.ylabel('Mean Residual', fontsize=14)
    plt.title(f'Mean residual evolution (assimilation up to t = {target_time:.2f})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'residual_evolution_t_{selected_time_s:.2f}s_ensemble_{N_ensemble}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
                                                    
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, spread_plot, 'purple', linewidth=2.5, label='Spread')
    plt.axvline(x=selected_time_s, color='red', linestyle='--', linewidth=2, 
                label=f'Target time (t={selected_time_s:.2f}s)')
    plt.xlabel('Time t (s)', fontsize=14)
    plt.ylabel('Spread', fontsize=14)
    plt.title(f'Spread evolution (assimilation up to t = {target_time:.2f})', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spread_evolution_t_{selected_time_s:.2f}s_ensemble_{N_ensemble}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
                           
                                      
                      
    
                              
                                                                  
                                              
                                        
                                                       
    
                                 
                                                                               
                                                
                                     
                                                                        
                                             
    
    
                        
                                                                                                    
                
    
                 
    plt.figure(figsize=(14, 8))
    
                               
    plt.plot(x, selected_truth, 'b-', linewidth=3, label='Truth', zorder=5)
    plt.plot(x, selected_est, 'r--', linewidth=2.5, label='EnKF', zorder=4)
    
                  
    lower_2sigma = selected_est - 2*selected_std
    upper_2sigma = selected_est + 2*selected_std
    
    plt.fill_between(x, lower_2sigma, upper_2sigma, 
                     alpha=0.4, color='red', zorder=2)
    
    plt.xlabel('Spatial Position x', fontsize=14)
    plt.ylabel('Wave Amplitude u', fontsize=14)
    plt.title(f'EnKF prediction with error bands (assimilation up to t = {target_time:.2f})', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_with_error_bands_at_t_{selected_time_s:.2f}s_ensemble_{N_ensemble}.png'), dpi=300, bbox_inches='tight')

                                          
    rmse_at_time = compute_rmse_over_domain(selected_truth, selected_est)
    rogue_threshold = 1.0
    rogue_region = selected_truth > rogue_threshold
    if not np.any(rogue_region):
        rogue_region = None
    peak_amp_error = compute_peak_amplitude_error(selected_truth, selected_est, rogue_region)
    mean_abs_err_peak_region = compute_mean_abs_error_region(selected_truth, selected_est, rogue_region)

    print(f"RMSE (domain) at t={selected_time_s:.2f}s: {rmse_at_time:.6f}")
    if rogue_region is None:
        print("no region detected")
    else:
        print(f"Mean absolute error in rogue region: {mean_abs_err_peak_region:.6f}")

                                    
    mse_evolution_csv = os.path.join(save_dir, f'mse_evolution_N_{N_ensemble}_Qd_{Q_d}_t_{selected_time_s:.2f}s.csv')
    import pandas as pd
    mse_data = pd.DataFrame({
        'time': time_vec[mask_upto],
        'mse': mse[mask_upto],
        'residual_mean': residual_mean[mask_upto],
        'spread': spread[mask_upto]
    })
    mse_data.to_csv(mse_evolution_csv, index=False)
    print(f"MSE evolution data saved to: {mse_evolution_csv}")
    
                         
    metrics_csv = os.path.join(save_dir, 'metrics_summary_ENKF.csv')
    metrics_row = {
        'target_time_s': float(selected_time_s),
        'run_target_time_s': float(target_time),
        'N_ensemble': int(N_ensemble),
        'Q_d': float(Q_d),
        'rmse_domain': rmse_at_time,
        'peak_amplitude_error': peak_amp_error,
        'mean_abs_error_peak_region': mean_abs_err_peak_region,
        'mse_evolution_file': f'mse_evolution_N_{N_ensemble}_Qd_{Q_d}_t_{selected_time_s:.2f}s.csv'
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
    plt.show()
    
    print(f"\nAll figures saved")
    
    print(f"Final MSE: {mse[-1]:.6f}")
    print(f"Final Mean Residual: {residual_mean[-1]:.6f}")
    print(f"Final Ensemble Spread: {spread[-1]:.6f}")
    print(f"Mean ensemble std: {np.mean(std[-1]):.4f}")

if __name__ == "__main__":
    create_all_figures(target_time=7.8, N_ensemble=150, Q_d=10, time_chosen=7.8)