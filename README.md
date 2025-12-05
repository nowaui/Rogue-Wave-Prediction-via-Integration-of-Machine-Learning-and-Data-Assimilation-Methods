# Rogue-Wave-Prediction-via-Integration-of-Machine-Learning-and-Data-Assimilation-Methods

This repository contains the source code accompanying the MSc thesis  
**“Rogue Wave Prediction via Integration of Machine Learning and Data Assimilation Methods”**

The code implements:
- A **mKdV** solver (finite differences + RK4) to generate rogue wave events.
- **CNN** and **UNet** surrogates for single-step and short-horizon multi-step prediction of the wave field.
- **Extended Kalman Filter (EKF)** and **Ensemble Kalman Filter (EnKF)** for data assimilation with the physical mKdV model.
- A **hybrid EnKF–UNet** system and **covariance localization** experiments based on a Gaspari–Cohn kernel.

**Important:** many scripts currently use **Windows absolute file paths** (e.g. `C:\Users\DELL\Desktop\...`). To run the code on your own machine, you **must update these paths** to your own local directory or convert them to **relative paths**

A typical workflow to reproduce the thesis experiments is:
### Adjust all file paths
1. Search for `C:\\Users\\DELL\\...` and similar patterns in all `.py` files.
2. Replace them with appropriate **relative paths** or your own absolute paths.

### Generate truth trajectories and ML datasets
1. Run `mKdV_RK4_create_dataset.py` to:
   - Generate a long high-resolution mKdV trajectory that contains a two-soliton collision leading to a rogue wave–like event.
   - Produce datasets for multiple two-soliton experiments.
2. Verify that the output data files are written to the expected directories.

### Train single-step surrogates

1. Run `CNN_final.py` to train the baseline 1D CNN model:
   - Input: current state `u_t`.
   - Target: increment `Δu_{t+1}`.
2. Run `Unet_final.py` to train the 1D UNet model:
   - Uses circular padding to enforce periodic boundary conditions.
   - Typically yields better performance around the rogue wave peak.
3. Check that:
   - Training and validation losses decrease as expected.
   - Rollout errors (e.g. RMSE over time) are reasonable for the forecast horizon.

### Train multi-step curriculum models

1. Run:
   - `Unetmulti_final_Max.py` 
   - `Unetmulti_final_Variable.py`,
   - `CNNmulti_final_Max.py`,
   - `CNNmulti_final_Variable.py`.
2. Use curriculum learning with horizons `m = 1, 2, 3, 4`:
   - First train for 1-step prediction.
   - Then extend to 2, 3, and 4 steps by reusing and fine-tuning the earlier weights.
3. Confirm that the correct checkpoint path is set in each script and that new `.pt` files are saved for the multi-step models.

### Run data assimilation with the physical model

1. Use the `ENKF_normal_*` scripts to perform EnKF with the mKdV solver:
   - Load or regenerate the truth trajectory.
   - Define the observation operator and observation noise.
   - Initialize an ensemble around a background state.
2. Run the filter over the time interval containing the rogue wave event.
3. Save and inspect:
   - Time series of RMSE and ensemble spread.
   - Snapshots of the ensemble mean vs. truth at key times.
   - Any additional plots produced by the scripts.

### Run hybrid EnKF–UNet experiments

1. Make sure that:
   - A trained UNet is available.
2. Run:
   - `ENKF_Unet_ensemble.py` for the baseline hybrid configuration.
   - `ENKF_Unet_effect_of_radius.py` and related scripts to:
     - Vary the localization radius.
     - Vary observation density.
     - Vary ensemble size.
3. Compare hybrid EnKF–UNet results against:
   - Pure physical-model EnKF runs.
   - Pure UNet rollouts (without data assimilation).

## Environment

All experiments were run with **Python 3.9.13**.

To install the required dependencies, run:

```bash
pip install -r requirements.txt





    
