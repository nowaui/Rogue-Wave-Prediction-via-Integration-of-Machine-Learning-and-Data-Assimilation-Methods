import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pandas as pd
from scipy.fft import fft, fftfreq

def F(u, N_x, dx, epsilon, mu):
    F = np.zeros(N_x)
    for i in range(2, N_x - 2):
        u_x = (u[i+1] - u[i-1]) / (2 * dx)
        u_xxx = (u[i+2] - 2 * u[i+1] + 2 * u[i-1] - u[i-2]) / (2 * dx**3)
        u_xx = (u[i+1] - 2 * u[i] + u[i-1]) / dx**2
        F[i] = -epsilon * u[i]**2 * u_x - mu * u_xxx + 0.01 * u_xx
    F[0:2] = F[N_x-4:N_x-2]
    F[N_x-2:N_x] = F[2:4]
    return F

# RK4
def rk4_step(u, N_x, dt, dx, epsilon, mu):
    k1 = F(u, N_x, dx, epsilon, mu)
    k2 = F(u + 0.5 * dt * k1, N_x, dx, epsilon, mu)
    k3 = F(u + 0.5 * dt * k2, N_x, dx, epsilon, mu)
    k4 = F(u + dt * k3, N_x, dx, epsilon, mu)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

L = 50.0
N_x = int(L / 0.1)
dx = 0.1
x = np.linspace(-L/2, L/2, N_x, endpoint=False)
epsilon = 6.0
mu = 1.0
T = 20.0
N_t = int(T / 0.0001)
dt = 0.0001
t0 = -5.0

save_csv = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/CSV"
save_ani = "C:/Users/DELL/Desktop/Thesis code/KdV code/Numerical sol/Ani"

# a = [0.8, -0.65]
x0 = [0.0, 0.0]
noise_scale = 0.0001
frames_to_save = 200  
step = N_t // frames_to_save

roguewave_counts = [15,30,45]

for num_exp in roguewave_counts:
    all_solutions_noise = []
    all_solutions_no_noise = []
    all_time_steps = []
    all_differences = []

    for i in range(num_exp):
        # np.random.seed(60)

        delta1 = np.random.uniform(0, 0.1)
        delta2 = np.random.uniform(0, 0.1)
        a1 = [1.0, -0.8]  # No noise
        a2 = [1.0 + delta1, -0.8 + delta2]  # With noise

        u0_no_noise = np.zeros(N_x)
        for j in range(len(a1)):
            u0_no_noise += a1[j] / np.cosh(a1[j] * (x - x0[j] - a1[j]**3 * t0))

        u0_noise = np.zeros(N_x)
        for j in range(len(a2)):
            u0_noise += a2[j] / np.cosh(a2[j] * (x - x0[j] - a2[j]**3 * t0))
        # noise = np.random.randn(N_x) * noise_scale
        # u0_noise = u0_noise + noise

        u_no_noise = u0_no_noise.copy()
        u_noise = u0_noise.copy()

        u_sol_no_noise = [u0_no_noise.copy()]
        u_sol_noise = [u0_noise.copy()]
        differences = [(u0_noise - u0_no_noise).copy()]
        time_steps = [t0]

        for n in range(N_t):
            u_no_noise = rk4_step(u_no_noise, N_x, dt, dx, epsilon, mu)
            u_noise = rk4_step(u_noise, N_x, dt, dx, epsilon, mu)
            if n % step == 0 or n == N_t - 1:
                u_sol_no_noise.append(u_no_noise.copy())
                u_sol_noise.append(u_noise.copy())

                differences.append(u_noise - u_no_noise)
                time_steps.append(t0 + (n + 1) * dt)

                print(f"Experiment {i + 1}/{num_exp}, t={time_steps[-1]:.2f}, max|u_noise|={np.max(np.abs(u_noise)):.3f}")

        all_solutions_no_noise.append(u_sol_no_noise)
        all_solutions_noise.append(u_sol_noise)
        all_differences.append(differences)
        all_time_steps.append(time_steps)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlim(-L/2, L/2)
        ax1.set_ylim(-2, 2)
        line1, = ax1.plot(x, u_sol_noise[0], lw=2, label='with noise')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u')
        ax1.grid(True)

        def update(frame):
            line1.set_ydata(u_sol_noise[frame])
            title = ax1.set_title(f'mKdV (t={time_steps[frame]:.2f}, exp={i+1})')
            return line1, title

        ani1 = FuncAnimation(fig1, update, frames=len(u_sol_noise), interval=100, blit=True)
        ani_path1 = os.path.join(save_ani, f"mKdV_two_solitons_noise_{num_exp}.mp4")
        ani1.save(ani_path1, writer='ffmpeg', dpi=200)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.set_xlim(-L/2, L/2)
        ax2.set_ylim(-0.5, 0.5)
        line2, = ax2.plot(x, differences[0], lw=2, color='red', label='Difference (noise - no noise)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('differences')
        ax2.grid(True)
        ax2.legend()

        def update_residual(frame):
            line2.set_ydata(differences[frame])
            title = ax2.set_title(f'mKdV Differences (t={time_steps[frame]:.2f}, exp={i+1})')
            return line2, title

        ani2 = FuncAnimation(fig2, update_residual, frames=len(differences), interval=100, blit=True)
        ani_path2 = os.path.join(save_ani, f"mKdV_two_solitons_amplitude_noise_{num_exp}_differences.mp4")
        ani2.save(ani_path2, writer='ffmpeg', dpi=200)
        plt.close(fig2)

        fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax3.set_xlim(-L/2, L/2)
        ax3.set_ylim(-2, 2)
        ax4.set_xlim(-L/2, L/2)
        ax4.set_ylim(-1e-2, 1e-2)  # Zoomed-in y-axis range
        line3, = ax3.plot(x, u_sol_noise[0], lw=2, color='blue', label='with noise')
        line4, = ax4.plot(x, differences[0], lw=2, color='red', label='Difference (noise - no noise)')
        ax3.set_ylabel('u')
        ax4.set_xlabel('x')
        ax4.set_ylabel('Difference')
        ax3.grid(True)
        ax4.grid(True)
        ax3.legend()
        ax4.legend()

        def update_combined(frame):
            line3.set_ydata(u_sol_noise[frame])
            line4.set_ydata(differences[frame])
            title = ax3.set_title(f'mKdV (t={time_steps[frame]:.2f}, exp={i+1})')
            return line3, line4, title

        ani3 = FuncAnimation(fig3, update_combined, frames=len(u_sol_noise), interval=100, blit=True)
        ani_path3 = os.path.join(save_ani, f"mKdV_two_solitons_amplitude_noise_{num_exp}_combined.mp4")
        ani3.save(ani_path3, writer='ffmpeg', dpi=200)
        plt.close(fig3)


        # frame_list = [10, 83, 190]
        # for frame in frame_list:
        #     diff = differences[frame]
        #     fft_result = fft(diff)
        #     freqs = fftfreq(N_x, dx)
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(freqs[:N_x//2], np.abs(fft_result)[:N_x//2], color='purple')
        #     plt.xlabel('Frequency')
        #     plt.ylabel('Amplitude')
        #     plt.title(f'Frequency Spectrum of Difference at t={time_steps[frame]:.2f}, exp={i+1}')
        #     plt.grid(True)
        #     fft_path = os.path.join(save_ani, f"fft_spectrum_diff_exp_{i+1}_t_{time_steps[frame]:.2f}.png")
        #     plt.savefig(fft_path, dpi=200)
        #     plt.close()

    data = {'x': x}
    for num in range(num_exp):
        for i, t in enumerate(all_time_steps[num]):
            data[f'u_t{t:.2f}_exp{num+1}'] = all_solutions_noise[num][i]
    df = pd.DataFrame(data)
    csv_path = os.path.join(save_csv, f"mKdV_two_solitons_noise_{num_exp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset with {num_exp} experiments to {csv_path}")

    combined_frames = []
    combined_titles = []
    for exp_idx in range(num_exp):
        for i in range(len(all_solutions_noise[exp_idx])):
            combined_frames.append(all_solutions_noise[exp_idx][i])
            combined_titles.append(f'mKdV (t={all_time_steps[exp_idx][i]:.2f}, exp={exp_idx+1})')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-1e-3, 1e-3)
    line, = ax.plot(x, combined_frames[0], lw=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid(True)

    def update_combined(frame):
        line.set_ydata(combined_frames[frame])
        title = ax.set_title(combined_titles[frame])
        return line, title

    ani = FuncAnimation(fig, update_combined, frames=len(combined_frames), interval=100, blit=True)
    ani_path = os.path.join(save_ani, f"mKdV_two_solitons_noise_{num_exp}_combined.mp4")
    ani.save(ani_path, writer='ffmpeg', dpi=200)
    plt.close(fig)

print('finished')

