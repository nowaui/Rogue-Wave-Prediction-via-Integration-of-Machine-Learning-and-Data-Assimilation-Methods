# Machine Learning

This folder contains the convolutional neural network (CNN) and UNet models used as data-driven surrogates for the mKdV dynamics, as well as their multi-step curriculum extensions.

## Scripts

### Single-step models

- `CNN_final.py`
  - Baseline 1D CNN.
  - Learns to predict the one-step increment `Δu_{t+1}` from the current state `u_t`.
  - Evaluation is done by rolling out the model over many steps and comparing to the numerical mKdV solution.

- `Unet_final.py`
  - 1D UNet with encoder–decoder structure and skip connections.
  - Uses **circular padding** to respect periodic boundary conditions.
  - Also predicts `Δu_{t+1}`, so that
    \[
    u_{t+1} = u_t + Δu_{t+1}.
    \]

### Multi-step (curriculum) models

- `CNNmulti_final_Max.py`
- `CNNmulti_final_Variable.py`
- `Unetmulti_final_Max.py`
- `Unetmulti_final_Variable.py`

These scripts extend the single-step models to **multi-step prediction** using curriculum learning with horizons  
`m = 1, 2, 3, 4`.

- **Max** variants: output head has fixed size, but only the first `m` channels are used at each stage.
- **Variable** variants: the size of the output head changes with `m`.

### Other utilities

- `compare.py`
  - Compare different models (e.g. CNN vs UNet) on the same test trajectories.

## Typical Usage

1. **Generate data**

   First run the script in `Numerical Simulation`

2. **Train a single-step model**
   
   Adjust hyperparameters and paths at the top of each script as needed.

3. **Train multi-step models**
  
4. **Evaluation and comparison**
    
   Use `compare.py` to produce rollouts and error curves, and to compare CNN vs UNet, single-step vs multi-step, etc.

   
