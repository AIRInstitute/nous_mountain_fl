# Flower Experiments for Mountain-Pass Beacons and EV Autonomy

This repository implements a proof of concept for the idea of **V2I beacons at mountain passes without external connectivity**. The beacon acts as a local federated server; each car that crosses the pass acts as an FL client. The goal is not to predict point by point, but to estimate **on entry** the energy required to cross the segment and compare it **on exit** with the actual energy consumed.

The central comparison is:

```text
Flat physical baseline without topography
vs
Residual MLP with topography trained centrally
vs
Residual MLP with topography trained with Flower/FedAvg
vs
Residual MLP with topography trained with Flower/FedProx
```

The main baseline is `B1_flat_physics_entry_no_topography`: it uses distance, nominal physical parameters, constant expected speed, rolling resistance, aerodynamics, and auxiliaries, but forces zero grade. It represents the case "I know how many km remain, but I don't know the pass's topography".

The proposed neural network does not replace the physics. It learns a residual:

```text
E_pred_segment = E_flat_physics_segment + MLP(topographic_features, weather, vehicle)
E_pred_pass    = sum(E_pred_segment)
```

## Structure

```text
configs/
  default.yaml                  # Base single-split experiments
  quick.yaml                    # Fast smoke test of run_all
  extended.yaml                 # Cross-validation + beacon learning curve
  quick_extended.yaml           # Fast smoke test of run_extended

data/
  raw/                          # Copy all valid CSVs here
  expected_files.txt            # Indicative list of expected files

src/mountain_pass_fl/
  data.py                       # Loading, metadata inference, segmentation
  baselines.py                  # Basic and nominal physical models
  features.py                   # Normalization and one-hot
  models.py                     # Residual MLP
  train.py                      # Centralized PyTorch training
  flower_exp.py                 # FedAvg/FedProx with Flower
  prequential.py                # Predict-then-update beacon evaluation
  cross_validation.py           # Cross-validation by routes/groups
  learning_curve.py             # Progressive beacon learning curve
  risk.py                       # Risk warnings and false-safe rate
  plotting.py                   # Figures
  run_all.py                    # Base orchestrator
  run_extended.py               # CV + learning curve orchestrator

scripts/
  setup_windows.ps1/.bat
  run_all_windows.ps1/.bat
  run_extended_windows.ps1/.bat
  run_extended_windows_no_flower.ps1/.bat
  run_extended_quick_windows.ps1/.bat
```

## Expected data

The CSVs must be in `data/raw/`. For the main evaluation, use only the complete routes.

Indicative list of complete routes:

```text
datos_grid_S01_tesla_SOC90_dry24.csv
datos_grid_S02_tesla_SOC90_wet10.csv
datos_grid_S03_tesla_SOC90_snow_neg5.csv
datos_grid_S04_tesla_SOC40_dry24.csv
datos_grid_S05_tesla_SOC40_wet10.csv
datos_grid_S06_tesla_SOC40_snow_neg5.csv
datos_grid_S07_audi_SOC90_dry24.csv
datos_grid_S08_audi_SOC90_wet10.csv
datos_grid_S09_audi_SOC90_snow_neg5.csv
datos_grid_S10_audi_SOC40_dry24.csv
datos_grid_S11_audi_SOC40_wet10.csv
datos_grid_S12_audi_SOC40_snow_neg5.csv
```

If you want to guard against partial routes, edit `configs/extended.yaml`:

```yaml
data_filters:
  min_route_distance_m: 8000
```

With `0`, no filter is applied.

## Installation on Windows PowerShell

From the project root folder, where `pyproject.toml`, `requirements.txt`, and `src/` are located:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

You can also use:

```powershell
.\scripts\setup_windows.ps1
```

If PowerShell blocks the `.ps1` files, use the equivalent `.bat` files.

## Base execution

Runs a single split, baselines, centralized, Flower, and prequential:

```powershell
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_all --data-dir data\raw --out-dir outputs --config configs\default.yaml --rebuild-segments
```

Without Flower:

```powershell
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_all --data-dir data\raw --out-dir outputs --config configs\default.yaml --rebuild-segments --skip-flower
```

## Extended execution: cross-validation + beacon curve

This is the recommended command for the final experiments:

```powershell
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_extended --data-dir data\raw --out-dir outputs_extended --config configs\extended.yaml --rebuild-segments
```

Or with a script:

```powershell
.\scripts\run_extended_windows.ps1
```

If you want to run everything except Flower, useful for fast debugging:

```powershell
.\scripts\run_extended_windows_no_flower.ps1
```

Fast smoke test:

```powershell
.\scripts\run_extended_quick_windows.ps1
```

## What `run_extended.py` adds

### 1. Cross-validation by routes/clients

It never splits by rows or by random segments. Each fold leaves out complete clients/CSVs.

Included protocols:

```text
leave_one_route_out
leave_one_scenario_group_out     # vehicle + condition: Tesla_dry, Audi_snow, etc.
leave_one_weather_out            # dry/wet/snow
leave_one_vehicle_out            # Tesla/Audi
repeated_random_client_split     # several train/val/test splits per client
```

Outputs:

```text
outputs_extended/cv/cv_route_predictions.csv
outputs_extended/cv/cv_fold_summary.csv
outputs_extended/cv/cv_summary.csv
outputs_extended/cv/cv_splits.csv
outputs_extended/cv/risk/risk_summary_grouped.csv
```

### 2. Progressive beacon learning curve

It simulates vehicle arrival orders:

```text
car enters -> predicts with current model
car exits -> updates the beacon
next car enters -> uses improved model
```

It is repeated with many arrival-order permutations and with different batch sizes:

```text
batch_size = 1, 2, 4
```

This lets you answer:

```text
What happens when the beacon has seen 0, 1, 2, 4, 8, or all cars?
```

Outputs:

```text
outputs_extended/learning_curve/learning_curve_route_predictions.csv
outputs_extended/learning_curve/learning_curve_point_summary.csv
outputs_extended/learning_curve/learning_curve_summary.csv
outputs_extended/learning_curve/risk/risk_summary_grouped.csv
```

`remaining_unseen` evaluates on cars that have not yet passed. It is the most honest metric.

`all_routes_replay` re-evaluates all routes with the model after seeing k cars. It serves as a diagnostic, not as pure generalization when k > 0.

### 3. Explicit Flower checkpoints

`learning_curve.py` includes a fast sequential simulation of FedAvg/FedProx. In addition, `run_extended.py` can launch checkpoints with real Flower:

```text
first 2 cars seen
first 4 cars seen
first 8 cars seen
all cars seen
```

Outputs:

```text
outputs_extended/learning_curve/flower_checkpoints/flower_checkpoints_route_predictions.csv
outputs_extended/learning_curve/flower_checkpoints/flower_checkpoints_summary.csv
```

These checkpoints are slower because each point starts a Flower/Ray simulation. They can be disabled with `--skip-flower`.

## Main outputs

```text
outputs_extended/segments.csv
outputs_extended/client_metadata.csv
outputs_extended/cv/cv_summary.csv
outputs_extended/learning_curve/learning_curve_summary.csv
outputs_extended/learning_curve/flower_checkpoints/flower_checkpoints_summary.csv
outputs_extended/plots/
outputs_extended/EXTENDED_REPORT.md
```

Relevant figures:

```text
outputs_extended/plots/cv_mae_boxplot_*.png
outputs_extended/plots/learning_curve_mae_remaining_unseen.png
outputs_extended/plots/learning_curve_false_safe_remaining_unseen.png
```

## Key metrics

```text
MAE total energy [kWh]
RMSE total energy [kWh]
Bias [Wh]
MAPE [%]
MAE final SOC [%]
false-safe rate
false-warning rate
risk recall
```

The metric most aligned with the paper's motivation is:

```text
false-safe rate = the system says "you can cross", but the vehicle actually ends up below the reserve
```

## Methodological note

The FL model does not receive `power_watts`, `current_a`, `voltage_v`, `energy_used_cum_wh`, or `energy_regen_cum_wh` as input. Those columns are only used to build the actual energy label. This prevents information leakage.

The neural network receives topographic and weather features and learns the residual with respect to the flat physical baseline. Therefore, the experimental argument is not "the NN replaces the physics", but rather:

```text
the beacon learns a local, pass-specific correction on top of a physical/general estimator.
```
