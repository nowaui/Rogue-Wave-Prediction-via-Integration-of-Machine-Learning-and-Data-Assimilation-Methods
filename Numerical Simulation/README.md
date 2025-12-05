# Numerical Simulation

This folder contains the numerical solver used to generate reference trajectories of the focusing modified Korteweg–de Vries (mKdV) equation. 
These trajectories are used both as “truth” for data assimilation and as training data for the machine learning surrogates.

> The script uses **absolute file paths** as they were configured on the original local machine.  
> Before running anything, you must update these paths to match your own directory structure (or convert them to relative paths).

## Overview

- `mKdV_RK4_create_dataset.py`  
  - Implements the 1D focusing mKdV equation with **periodic boundary conditions**.
  - Uses a **finite-difference spatial discretisation** and a **fourth-order Runge–Kutta (RK4)** time integrator.
  - Simulates **two-soliton collisions** that generate rogue wave–like events.
  - Generates:
    - A long high-resolution “truth” trajectory on a uniform spatial grid.
    - Datasets for CNN/UNet models.

---

## Usage

1. **Edit paths**

   Open `mKdV_RK4_create_dataset.py` and search for paths like:

   ```python
   truth_dir = r"C:\Users\DELL\Desktop\..."
   ```
  Replace them with locations that exist on your system
