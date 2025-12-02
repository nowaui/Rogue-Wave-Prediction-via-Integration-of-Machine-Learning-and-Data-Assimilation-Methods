import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
import os
import pandas as pd

class mKdVSystem:
    """mKdV system"""
    
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
    
    def F(self, u):
        u = np.clip(u, -3.0, 3.0)
        
        M = self.M
        F = np.zeros(M)
        
        u_ext = np.zeros(M + 4)
        u_ext[2:M+2] = u
        u_ext[0:2] = u[M-2:M]
        u_ext[M+2:M+4] = u[0:2]
        
        for m in range(M):
            m_ext = m + 2
            nonlinear = -self.epsilon * u_ext[m_ext]**2 * (u_ext[m_ext+1] - u_ext[m_ext-1]) * self.inv_2dx
            linear = -self.mu * (u_ext[m_ext+2] - 2*u_ext[m_ext+1] + 2*u_ext[m_ext-1] - u_ext[m_ext-2]) * self.inv_2dx3
                                                                                                     
            F[m] = nonlinear + linear
                                                   
        
        F = np.clip(F, -1000.0, 1000.0)
        return F
    
    def RK4_step(self, u):
        k1 = self.F(u)
        k2 = self.F(u + 0.5 * self.dt * k1)
        k3 = self.F(u + 0.5 * self.dt * k2)
        k4 = self.F(u + self.dt * k3)
        u_new = u + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        u_new = np.clip(u_new, -3.0, 3.0)
        return u_new
    
    def compute_jacobian(self, u):
        u = np.clip(u, -3.0, 3.0)
        
        M = self.M
        
        u_ext = np.zeros(M + 4)
        u_ext[2:M+2] = u
        u_ext[0:2] = u[M-2:M]
        u_ext[M+2:M+4] = u[0:2]
        
        row_indices = []
        col_indices = []
        data = []
        
        for m in range(M):
            m_ext = m + 2
            
                           
            J_mm = -12 * self.epsilon * u_ext[m_ext] * (u_ext[m_ext+1] - u_ext[m_ext-1]) * self.inv_2dx
            J_mm = np.clip(J_mm, -3000.0, 3000.0)  
            row_indices.append(m)
            col_indices.append(m)
            data.append(J_mm)
            
                            
            m_plus_1 = (m + 1) % M
            J_m_mp1 = -6 * self.epsilon * u_ext[m_ext]**2 * self.inv_2dx + 2 * self.mu * self.inv_2dx3
            J_m_mp1 = np.clip(J_m_mp1, -2000.0, 2000.0)  
            row_indices.append(m)
            col_indices.append(m_plus_1)
            data.append(J_m_mp1)
            
            m_minus_1 = (m - 1) % M
            J_m_mm1 = 6 * self.epsilon * u_ext[m_ext]**2 * self.inv_2dx - 2 * self.mu * self.inv_2dx3
            J_m_mm1 = np.clip(J_m_mm1, -2000.0, 2000.0) 
            row_indices.append(m)
            col_indices.append(m_minus_1)
            data.append(J_m_mm1)
            
            m_plus_2 = (m + 2) % M
            J_m_mp2 = -self.mu * self.inv_2dx3
            J_m_mp2 = np.clip(J_m_mp2, -1000.0, 1000.0)
            row_indices.append(m)
            col_indices.append(m_plus_2)
            data.append(J_m_mp2)
            
            m_minus_2 = (m - 2) % M
            J_m_mm2 = self.mu * self.inv_2dx3
            J_m_mm2 = np.clip(J_m_mm2, -1000.0, 1000.0)
            row_indices.append(m)
            col_indices.append(m_minus_2)
            data.append(J_m_mm2)
        
        return sp.csr_matrix((data, (row_indices, col_indices)), shape=(M, M))
    
class mKdVEKF:
    """Extended Kalman Filter"""
    
    def __init__(self, system, Q_std=1e-6, R_std=0.005, Q_d=5):
        self.system = system
        self.M = system.M
        self.Q = create_gaussian_Q(self.M, Q_std, Q_d)
        self.R_std = R_std
    
    def predict(self, u_est, P_est):
        u_est = np.clip(u_est, -3.0, 3.0)

        J = self.system.compute_jacobian(u_est)

        if hasattr(J, 'toarray'):
            J = J.toarray()
 
        J = np.clip(J, -3000.0, 3000.0)
        

        I = np.eye(self.M)
        F_k = I + self.system.dt * J
        
        u_pred = F_k @ u_est
        P_pred = F_k @ P_est @ F_k.T + self.Q

        u_pred = np.clip(u_pred, -3.0, 3.0)

        P_pred = P_pred + 1e-8 * np.eye(self.M)
        
        return u_pred, P_pred, F_k
    
    def update(self, u_pred, P_pred, y_obs, H, R):

        if hasattr(H, 'toarray'):
            H = H.toarray()
        if hasattr(R, 'toarray'):
            R = R.toarray()

        u_pred = np.clip(u_pred, -3.0, 3.0)
            
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        v = y_obs - H @ u_pred
        v = np.clip(v, -2.0, 2.0)
        
        u_est = u_pred + K @ v
        
        I_KH = np.eye(self.M) - K @ H
        P_est = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        
        u_est = np.clip(u_est, -3.0, 3.0)

        P_est = P_est + 1e-8 * np.eye(self.M)
                
        return u_est, P_est, K, v

def create_gaussian_Q(M, Q_std, Q_d):
    Q = np.zeros((M, M))
    
    for i in range(M):
        for j in range(M):
            dist = min(abs(i - j), M - abs(i - j))
            Q[i, j] = Q_std**2 * np.exp(-dist**2 / (2 * Q_d**2))
    
    print(Q[:5, :5])

    return Q

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


def predict_rogue_wave(target_time, Q_d, time_chosen=None):
    L = 50.0
    dx = 0.1
    M = int(L / dx)

    dt = 1e-4  
    T = target_time + 0.1  
    N_t = int(T / dt)
    x = np.linspace(-L/2, L/2, M, endpoint=False)
    
    print(f"t = {target_time:.2f}s")
    print(f"dt={dt}, T={T}, N_t={N_t}")
    print(f"L={L}, M={M}, dx={dx:.3f}")
    print(f"Q distance = {Q_d}")
    
    if target_time < 0 or target_time > T:
        print(f"Not in available time interval")
        return
    
    start_time = time.time()
    target_idx = int(target_time / dt)
    print(f"target index: {target_idx}")

    system = mKdVSystem(M, dx, dt)

    def two_soliton_ic(x, a=[1.0, -0.8], x0=[0.0, 0.0], t0=-5.0):

        u0 = np.zeros_like(x)
        for i in range(len(a)):
            u0 += a[i] / np.cosh(a[i] * (x - x0[i] - a[i]**3 * t0))
        return u0

                     
    u0_truth = two_soliton_ic(x)
    print(f"Initial condition: max|u| = {np.max(np.abs(u0_truth)):.3f}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(x, u0_truth, 'b-', linewidth=2, label='Two-Soliton Initial Condition')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('mKdV Two-Soliton Initial Condition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


    u_truth = np.zeros((N_t + 1, M))
    u_truth[0] = u0_truth
    
    u = u0_truth.copy()
    for k in range(N_t):
        u = system.RK4_step(u)
        u_truth[k+1] = u
        if (k+1) % 4000 == 0:
            print(f"{(k+1)/N_t*100:.0f}%, max|u| = {np.max(np.abs(u)):.3f}")
    
                                     
    selected_time_s = float(time_chosen) if time_chosen is not None else float(target_time)
    analysis_idx = int(round(selected_time_s / dt))
    analysis_idx = max(0, min(analysis_idx, N_t))
    analysis_time = analysis_idx * dt
    
    max_amplitudes = np.max(np.abs(u_truth), axis=1)
    print(f"\nAnalysis_time: t = {analysis_time:.2f}s, Amplitudes = {max_amplitudes[analysis_idx]:.3f}")

    obs_ratio= 0.75
    obs_interval = 3  
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
    
    print("Zero IC")
    ekf = mKdVEKF(system, Q_std=1e-3, R_std=obs_noise_std, Q_d=Q_d)
    
    u0_est = np.zeros(M) 
    P0 = 0.1 * np.eye(M) 
    
    u_est_history = np.zeros((N_t + 1, M))
    u_est_history[0] = u0_est
    mse_history = np.zeros(N_t + 1)
    mse_history[0] = np.mean((u0_est - u_truth[0])**2)
    std_history = np.zeros((N_t + 1, M))
    std_history[0] = np.sqrt(np.maximum(np.diag(P0), 0.0))
    
    u_est = u0_est.copy()
    P_est = P0.copy()
    
    if 0 in observations:
        y_obs = observations[0]
        u_est, P_est, _, _ = ekf.update(u_est, P_est, y_obs, H, R)
        u_est_history[0] = u_est
        mse_history[0] = np.mean((u_est - u_truth[0])**2)
        std_history[0] = np.sqrt(np.maximum(np.diag(P_est), 0.0))
    
    obs_count = 0
    for k in range(N_t):
        if (k+1) % 2000 == 0:
            print(f"  EKF process: {(k+1)/N_t*100:.0f}%, MSE = {mse_history[k]:.6f}, max|u| = {np.max(np.abs(u_est)):.3f}")
        
                 
        u_pred, P_pred, F_k = ekf.predict(u_est, P_est)

        if np.any(np.isnan(u_pred)) or np.any(np.isinf(u_pred)):
            print(f"Unstable")
            break
        
                
        if (k+1) in observations:
            y_obs = observations[k+1]
            u_est, P_est, K, v = ekf.update(u_pred, P_pred, y_obs, H, R)
            obs_count += 1
        else:
            u_est = u_pred
            P_est = P_pred
        
        u_est_history[k+1] = u_est
        mse_history[k+1] = np.mean((u_est - u_truth[k+1])**2)
        std_history[k+1] = np.sqrt(np.maximum(np.diag(P_est), 0.0))
        
    
    print(f"\nFinal MSE: {mse_history[-1]:.6f}")
    print(f"Number of observation {obs_count}")
    

    save_dir = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\Final code\Final_Final_code\Data_assimilation"

    mse_at_analysis = mse_history[analysis_idx]
    max_error = np.max(np.abs(u_truth[analysis_idx] - u_est_history[analysis_idx]))
    time_vec = np.arange(N_t + 1) * dt
    valid_mse = np.isfinite(mse_history)
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    ax1.plot(x, u_truth[analysis_idx], 'b-', linewidth=2.5, label='True Wave')
    ax1.plot(x, u_est_history[analysis_idx], 'r--', linewidth=2.5, label='EKF Prediction')

    
    ax1.set_xlabel('Spatial Position x', fontsize=14)
    ax1.set_ylabel('Wave Amplitude u', fontsize=14)
    ax1.set_title(f'Wave prediction (assimilation up to t = {target_time:.2f})', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    
                                                        
    
    plt.tight_layout()
    
    save_path1 = os.path.join(save_dir, f"Wave_prediction_t{analysis_time:.2f}Q_d{Q_d}.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"Fig1 saved {save_path1}")
    plt.show()
    

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))

                                              
    mask_upto = time_vec <= (analysis_time + 1e-12)
    if np.any(valid_mse & mask_upto):
        ax2.semilogy(time_vec[valid_mse & mask_upto], mse_history[valid_mse & mask_upto], 'g-', linewidth=2.5)
    else:
        ax2.plot(time_vec[mask_upto], np.ones(np.sum(mask_upto)), 'g-', linewidth=2.5, label='MSE (No Valid Values)')
    
                        
    ax2.axvline(x=analysis_time, color='red', linestyle='--', linewidth=3, label=f'Analysis Time (t={analysis_time:.2f}s)')
    
    ax2.set_xlabel('Time t (s)', fontsize=14)
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=14)
    ax2.set_title(f'Prediction error evolution (assimilation up to t = {target_time:.2f})', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    
    plt.tight_layout()

    save_path2 = os.path.join(save_dir, f"MSE_evolution_t{analysis_time:.2f}s_Q_d{Q_d}.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Fig2 saved: {save_path2}")
    plt.show()


    fig3, (ax1_comb, ax2_comb) = plt.subplots(2, 1, figsize=(12, 10))

    ax1_comb.plot(x, u_truth[analysis_idx], 'b-', linewidth=2.5, label='True Wave')
    ax1_comb.plot(x, u_est_history[analysis_idx], 'r--', linewidth=2.5, label='EKF Prediction')
    lower_2sigma = u_est_history[analysis_idx] - 2 * std_history[analysis_idx]
    upper_2sigma = u_est_history[analysis_idx] + 2 * std_history[analysis_idx]
    ax1_comb.fill_between(x, lower_2sigma, upper_2sigma,
                         alpha=0.4, color='red', label='±2σ band')
    
    ax1_comb.set_xlabel('Spatial Position x', fontsize=12)
    ax1_comb.set_ylabel('Wave Amplitude u', fontsize=12)
    ax1_comb.set_title(f'Wave prediction (assimilation up to t = {target_time:.2f})', 
                      fontsize=14, fontweight='bold')
    ax1_comb.legend(fontsize=12)
    ax1_comb.grid(True, alpha=0.3)
    
    ax1_comb.text(0.02, 0.98, f'MSE = {mse_at_analysis:.6f}', 
                 transform=ax1_comb.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=11)

    if np.any(valid_mse):
        ax2_comb.semilogy(time_vec[valid_mse], mse_history[valid_mse], 'g-', linewidth=2)
    else:
        ax2_comb.plot(time_vec, np.ones_like(time_vec), 'g-', linewidth=2, label='MSE (No Valid Values)')
       
    ax2_comb.set_xlabel('Time t (s)', fontsize=12)
    ax2_comb.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax2_comb.set_title(f'Prediction error evolution (assimilation up to t = {target_time:.2f})', fontsize=14, fontweight='bold')
    ax2_comb.grid(True, alpha=0.3)
    ax2_comb.legend(fontsize=12)

    plt.tight_layout()
    
    save_path3 = os.path.join(save_dir, f"Combined_plots_t{analysis_time:.2f}Q_d{Q_d}.png")   
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"Fig3 saved: {save_path3}")
    plt.show()
    
                                      
    print(f"Analysis time: t = {analysis_time:.2f}s")
    print(f"Prediction accuracy: {np.max(np.abs(u_est_history[analysis_idx]))/np.max(np.abs(u_truth[analysis_idx]))*100:.1f}%")
    print(f"MSE at analysis time: {mse_at_analysis:.6f}")
    running_time = time.time() - start_time
    print(f"Completed in {running_time:.2f}s")

                                          
    selected_truth = u_truth[analysis_idx]
    selected_est = u_est_history[analysis_idx]
    rmse_at_time = compute_rmse_over_domain(selected_truth, selected_est)
    rogue_threshold = 1.0
    rogue_region = selected_truth > rogue_threshold
    if not np.any(rogue_region):
        rogue_region = None
    peak_amp_error = compute_peak_amplitude_error(selected_truth, selected_est, rogue_region)
    mean_abs_err_peak_region = compute_mean_abs_error_region(selected_truth, selected_est, rogue_region)

    print(f"RMSE (domain) at t={analysis_time:.2f}s: {rmse_at_time:.6f}")
    if rogue_region is None:
        print("no region detected")
    else:
        print(f"Mean absolute error in rogue region: {mean_abs_err_peak_region:.6f}")

    metrics_csv = os.path.join(save_dir, 'metrics_summary_EKF.csv')
    metrics_row = {
        'target_time_s': float(analysis_time),
        'run_target_time_s': float(target_time),
        'Q_d': float(Q_d),
        'rmse_domain': rmse_at_time,
        'peak_amplitude_error': peak_amp_error,
        'mean_abs_error_peak_region': mean_abs_err_peak_region,
        'computational_time_s': float(running_time)
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

if __name__ == "__main__":
    predict_rogue_wave(target_time=7.8, Q_d=10, time_chosen=7.8)