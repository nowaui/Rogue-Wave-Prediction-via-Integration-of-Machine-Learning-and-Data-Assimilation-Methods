import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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


def compute_prediction_metrics(truth, estimate, rogue_threshold=1.0):
    region = truth > rogue_threshold
    if not np.any(region):
        region = None
    rmse = compute_rmse_over_domain(truth, estimate)
    peak_err = compute_peak_amplitude_error(truth, estimate, region)
    mean_abs_peak = compute_mean_abs_error_region(truth, estimate, region)
    return {
        "rmse": rmse,
        "peak_amplitude_error": peak_err,
        "mean_abs_error_peak_region": mean_abs_peak
    }


from CNN_final import CNN as CNN_single
from CNNmulti_final_Max import CNN as CNN_multi_max
from CNNmulti_final_Variable import CNN as CNN_multi_variable
from Unet_final import UNetModel as UNet_single
from Unetmulti_final_Max import UNetModel as UNet_multi_max
from Unetmulti_final_Variable import UNetModel as UNet_multi_variable


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = r"C:\Users\DELL\Desktop\Thesis code\KdV code\Numerical sol\CSV\mKdV_two_solitons_noise_1.csv"
SAVE_DIR = os.path.join(BASE_DIR, "compare_new")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EXPERIMENTS = 1
HISTORY_LENGTH = 64  
CNN_MULTI_HISTORY_LENGTH = 64  
UNET_MULTI_HISTORY_LENGTH = HISTORY_LENGTH
BLOCK_STEPS_CNN_MULTI = 4
BLOCK_STEPS_UNET_MULTI = 4
CNN_MULTI_M_MAX = 4  
UNET_MULTI_M_MAX = BLOCK_STEPS_UNET_MULTI
DATA_BLOCK_STEPS = max(BLOCK_STEPS_CNN_MULTI, BLOCK_STEPS_UNET_MULTI)
TARGET_STEP = DATA_BLOCK_STEPS - 1  
TARGET_TIME_INDEX = 10  

MODEL_PATHS = {
    "CNN_single": os.path.join(BASE_DIR, "CNNmodels_final", "model_batch_32_exp45.pth"),
    "UNet_single": os.path.join(BASE_DIR, "Unetmodels_final", "model_batch_128_exp45.pth"),
    "CNN_multi_Max_Output": os.path.join(BASE_DIR, "CNNmodels_multifinal_v2", "model_batch_128_exp45_k64_m4.pth"),
    "CNN_multi_Variable_Output": os.path.join(BASE_DIR, "CNNmodels_multifinal_v3", "model_batch_128_exp30_k64_m4.pth"),
    "UNet_multi_Max_Output": os.path.join(BASE_DIR, "Unetmodels_multifinal_v2", "model_batch_32_exp15_k64_m4.pth"),
    "UNet_multi_Variable_Output": os.path.join(BASE_DIR, "Unetmodels_multifinal_v3", "model_batch_64_exp45_k64_m4.pth"),
}

MODEL_CONFIGS = [
    {
        "name": "CNN_single",
        "label": "CNN_single",
        "type": "single",
        "builder": CNN_single,
        "builder_kwargs": {}
    },
    {
        "name": "UNet_single",
        "label": "UNet_single",
        "type": "single",
        "builder": UNet_single,
        "builder_kwargs": {}
    },
    {
        "name": "CNN_multi_Max_Output",
        "label": "CNN_multi_Max_Output",
        "type": "multi_fixed",
        "builder": CNN_multi_max,
        "builder_kwargs": {
            "k": CNN_MULTI_HISTORY_LENGTH,
            "m": BLOCK_STEPS_CNN_MULTI,
            "m_max": CNN_MULTI_M_MAX
        },
        "block_steps": BLOCK_STEPS_CNN_MULTI,
        "history_length": CNN_MULTI_HISTORY_LENGTH,
        "variable_mode": "fixed"
    },
    {
        "name": "CNN_multi_Variable_Output",
        "label": "CNN_multi_Variable_Output",
        "type": "multi_variable",
        "builder": CNN_multi_variable,
        "builder_kwargs": {
            "k": CNN_MULTI_HISTORY_LENGTH,
            "m": BLOCK_STEPS_CNN_MULTI,
            "m_max": CNN_MULTI_M_MAX
        },
        "block_steps": BLOCK_STEPS_CNN_MULTI,
        "history_length": CNN_MULTI_HISTORY_LENGTH,
        "variable_mode": "arg"
    },
    {
        "name": "UNet_multi_Max_Output",
        "label": "UNet_multi_Max_Output",
        "type": "multi_fixed",
        "builder": UNet_multi_max,
        "builder_kwargs": {
            "k": UNET_MULTI_HISTORY_LENGTH,
            "m": BLOCK_STEPS_UNET_MULTI,
            "m_max": UNET_MULTI_M_MAX
        },
        "block_steps": BLOCK_STEPS_UNET_MULTI,
        "history_length": UNET_MULTI_HISTORY_LENGTH,
        "variable_mode": "fixed"
    },
    {
        "name": "UNet_multi_Variable_Output",
        "label": "UNet_multi_Variable_Output",
        "type": "multi_variable",
        "builder": UNet_multi_variable,
        "builder_kwargs": {
            "k": UNET_MULTI_HISTORY_LENGTH,
            "m": BLOCK_STEPS_UNET_MULTI,
            "m_values": list(range(1, BLOCK_STEPS_UNET_MULTI + 1))
        },
        "block_steps": BLOCK_STEPS_UNET_MULTI,
        "history_length": UNET_MULTI_HISTORY_LENGTH,
        "variable_mode": "attr"
    },
]


def load_dataset(csv_path, num, k, m):
    data = pd.read_csv(csv_path)
    x = data["x"].values
    u_data = data.drop("x", axis=1).values.T.astype(np.float32)

    u_min, u_max = u_data.min(), u_data.max()
    u_normalized = 2 * (u_data - u_min) / (u_max - u_min) - 1

    trajectory_len = u_normalized.shape[0] // num
    X_list, y_list = [], []

    for i in range(num):
        start = i * trajectory_len
        end = (i + 1) * trajectory_len
        u_exp = u_normalized[start:end]

        for j in range(len(u_exp) - k - m + 1):
            X_sample = u_exp[j:j + k]
            deltas = []
            for t in range(m):
                delta = u_exp[j + k + t] - u_exp[j + k + t - 1]
                deltas.append(delta)
            X_list.append(X_sample.astype(np.float32))
            y_list.append(np.stack(deltas, axis=0).astype(np.float32))

    X_total = np.stack(X_list, axis=0)
    y_total = np.stack(y_list, axis=0)
    return X_total, y_total, x, u_min, u_max


def denormalize(u_norm, u_min, u_max):
    return (u_norm + 1.0) * (u_max - u_min) / 2.0 + u_min


def reconstruct_true_state(X_sample, y_sample, target_step):
    base_state = X_sample[-1]
    cumulative_delta = np.sum(y_sample[:target_step + 1], axis=0)
    return base_state + cumulative_delta


def deltas_to_states(base_state: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    states = []
    current = base_state.copy()
    for delta in deltas:
        current = current + delta
        states.append(current.copy())
    return np.array(states)

def build_and_load_model(config):
    path = MODEL_PATHS.get(config["name"])
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Model path for {config['name']} not found. Please update MODEL_PATHS.")

    model = config["builder"](**config["builder_kwargs"])
    state_dict = torch.load(path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Warning] Loading {config['name']} had missing keys {missing} and unexpected keys {unexpected}.")
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_state(model, config, X_sample, target_step):
    if config["type"] == "single":
        states = rollout_single(model, X_sample[-1], target_step + 1, DEVICE)
        return states[target_step]

    mode = config.get("variable_mode", "fixed")
    block_size = config.get("block_steps", DATA_BLOCK_STEPS)
    history_length = config.get("history_length", HISTORY_LENGTH)
    history_for_model = X_sample[-history_length:]
    states = rollout_multi_block(
        model,
        history_for_model,
        target_step + 1,
        block_size=block_size,
        mode=mode,
        history_length=history_length,
    )
    return states[target_step]

def rollout_single(model: torch.nn.Module, base_state: np.ndarray, num_steps: int, device: torch.device) -> np.ndarray:
    states = []
    current = base_state.copy()
    for _ in range(num_steps):
        inp = torch.tensor(current[np.newaxis, np.newaxis, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            delta = model(inp).cpu().numpy().squeeze()
        current = current + delta
        states.append(current.copy())
    return np.array(states)


def rollout_multi_block(model, history: np.ndarray, num_steps: int, block_size: int, mode: str, history_length: int) -> np.ndarray:
    history_buffer = history.copy()
    states = []
    steps_done = 0

    while steps_done < num_steps:
        steps_to_predict = min(block_size, num_steps - steps_done)
        inp = torch.tensor(history_buffer[np.newaxis, :, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            if mode == "arg":
                deltas = model(inp, steps_to_predict).detach().cpu().numpy()[0]
            elif mode == "attr":
                previous_m = getattr(model, "m", None)
                if not hasattr(model, "final_convs") or str(steps_to_predict) not in model.final_convs:
                    raise ValueError(f"Model does not support m={steps_to_predict}")
                model.m = steps_to_predict
                deltas = model(inp).detach().cpu().numpy()[0]
                model.m = previous_m
            else:
                block_deltas = model(inp).detach().cpu().numpy()[0]
                deltas = block_deltas[:steps_to_predict]

        base_state = history_buffer[-1]
        block_states = deltas_to_states(base_state, deltas)
        states.extend(block_states)

        history_buffer = np.concatenate([history_buffer, block_states], axis=0)
        history_buffer = history_buffer[-history_length:]
        steps_done += steps_to_predict

    return np.array(states)


def plot_pairwise_comparison(x, true_state, label_a, pred_a, rmse_a, label_b, pred_b, rmse_b, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(x, true_state, color="black", linewidth=2.5, label="True")
    axes[0].plot(x, pred_a, linewidth=2, label=f"{label_a}")
    axes[0].plot(x, pred_b, linewidth=2, label=f"{label_b}")
    axes[0].set_ylabel("u")
    axes[0].set_title(
        f"Prediction vs True ({label_a} RMSE={rmse_a:.4f}, {label_b} RMSE={rmse_b:.4f})"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.0, color="gray", linewidth=1, linestyle="--")
    axes[1].plot(x, pred_a - true_state, linewidth=2, label=f"{label_a} - True")
    axes[1].plot(x, pred_b - true_state, linewidth=2, label=f"{label_b} - True")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Difference")
    axes[1].set_title("Prediction - Truth")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    axes[1].legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_side_by_side_models(x, true_state, model_labels, predictions, rmse_summary, save_path):
    """Plot multiple models side-by-side against the truth with RMSE in titles."""
    if not model_labels:
        return

    fig, axes = plt.subplots(1, len(model_labels), figsize=(6 * len(model_labels), 4), sharey=True)
    if len(model_labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, model_labels):
        pred = predictions[label]
        rmse = rmse_summary[label]
        ax.plot(x, true_state, color="black", linewidth=2.5, label="True")
        ax.plot(x, pred, color="tab:blue", linestyle="--", linewidth=2, label=label)
        ax.set_title(f"{label} prediction (RMSE={rmse:.4f})")
        ax.set_xlabel("x")
        if ax is axes[0]:
            ax.set_ylabel("u")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    X_total, y_total, x, u_min, u_max = load_dataset(
        CSV_PATH,
        num=NUM_EXPERIMENTS,
        k=HISTORY_LENGTH,
        m=DATA_BLOCK_STEPS
    )

    inferred_sample = max(0, TARGET_TIME_INDEX - DATA_BLOCK_STEPS)
    sample_idx = min(inferred_sample, len(X_total) - 1)
    actual_state_index = sample_idx + DATA_BLOCK_STEPS

    if actual_state_index != TARGET_TIME_INDEX:
        print(f"[Info] Requested time index {TARGET_TIME_INDEX} mapped to available index {actual_state_index} "
              f"(window starts at sample {sample_idx}).")
    X_sample = X_total[sample_idx]
    y_sample = y_total[sample_idx]

    true_state_norm = reconstruct_true_state(X_sample, y_sample, TARGET_STEP)
    true_state = denormalize(true_state_norm, u_min, u_max)

    import time
    predictions = {}
    for config in MODEL_CONFIGS:
        start_time = time.time()
        model = build_and_load_model(config)
        pred_norm = predict_state(model, config, X_sample, TARGET_STEP)
        elapsed = time.time() - start_time
        predictions[config["label"]] = denormalize(pred_norm, u_min, u_max)
        print(f"[Timing] {config['label']}: prediction completed in {elapsed:.2f} s")

    metrics_summary = {}
    rmse_summary = {}
    for label, pred in predictions.items():
        metrics = compute_prediction_metrics(true_state, pred)
        metrics_summary[label] = metrics
        rmse_summary[label] = metrics["rmse"]

    for label_a, label_b in itertools.combinations(predictions.keys(), 2):
        filename = (
            f"{label_a}_vs_{label_b}_k{HISTORY_LENGTH}_m{DATA_BLOCK_STEPS}_step{TARGET_STEP + 1}"
            f"_time{actual_state_index}_sample{sample_idx + 1}.png"
        )
        save_path = os.path.join(SAVE_DIR, filename)
        side_filename = (
            f"{label_a}_vs_{label_b}_side_by_side_k{HISTORY_LENGTH}_m{DATA_BLOCK_STEPS}_step{TARGET_STEP + 1}"
            f"_time{actual_state_index}_sample{sample_idx + 1}.png"
        )
        side_save_path = os.path.join(SAVE_DIR, side_filename)
        plot_pairwise_comparison(
            x,
            true_state,
            label_a,
            predictions[label_a],
            rmse_summary[label_a],
            label_b,
            predictions[label_b],
            rmse_summary[label_b],
            save_path
        )
        plot_side_by_side_models(
            x,
            true_state,
            [label_a, label_b],
            predictions,
            rmse_summary,
            side_save_path
        )
        print(f"Saved pairwise figure: {save_path}")
        print(f"Saved side-by-side figure: {side_save_path}")

    single_model_labels = [label for label in ("CNN_single", "UNet_single") if label in predictions]
    if single_model_labels:
        single_filename = (
            f"single_model_comparison_k{HISTORY_LENGTH}_m{DATA_BLOCK_STEPS}_step{TARGET_STEP + 1}"
            f"_time{actual_state_index}_sample{sample_idx + 1}.png"
        )
        single_save_path = os.path.join(SAVE_DIR, single_filename)
        plot_side_by_side_models(
            x,
            true_state,
            single_model_labels,
            predictions,
            rmse_summary,
            single_save_path
        )
        print(f"Saved single-model comparison figure: {single_save_path}")

    print("\n=== RMSE Summary (k={}, m={}, target index={}) ===".format(HISTORY_LENGTH, DATA_BLOCK_STEPS, actual_state_index))
    print("{:30s} | {:>12s}".format("Model", "RMSE"))
    separator_line = "-" * 46
    print(separator_line)
    for label, metrics in sorted(metrics_summary.items(), key=lambda x: x[1]["rmse"]):
        rmse = metrics["rmse"]
        peak_err = metrics["peak_amplitude_error"]
        mean_peak = metrics["mean_abs_error_peak_region"]
        print(f"{label:30s} | {rmse:12.6e} | PeakAmpErr={peak_err:12.6e} | MeanAbsErrPeak={mean_peak:12.6e}")


if __name__ == "__main__":
    main()
