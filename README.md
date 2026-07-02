# EV Energy Simulation at the Puerto de la Quesera — Experimental Dataset

This repository contains a factorial experimental dataset produced by simulating a digital twin of an electric vehicle (EV) traversing a representative stretch of the **Puerto de la Quesera** (Sistema Central, GU-186 road). The dataset comprises 12 scenarios resulting from the factorial combination of vehicle, initial state of charge (SOC), and weather condition.

The pipeline is **DEM → MATLAB → RoadRunner → CARLA Sim 0.9.16 → energy model in Python**, with vehicle kinematic traces fed into a longitudinal energy model with weather coupling (HVAC, rolling resistance, battery capacity, regenerative braking).

This data supports a manuscript in preparation for *Open Research Europe (ORE)*, reporting the digital-twin methodology and the case-study results.

---

## 1. Geographic and topographic context

| Property | Value |
|---|---|
| Region | Sistema Central (Spain), border between Guadalajara and Segovia |
| Road | GU-186 (OSM way `157020458`, classified as *tertiary*) |
| Reference name | Puerto de la Quesera |
| Coordinate system | WGS84 / EPSG:4326 |
| Bounding box (DEM) | lon `[-3.4001°, -3.3540°]`, lat `[41.1610°, 41.1821°]` |
| DEM dimensions | 3.86 km × 2.35 km (9.08 km²) |
| DEM source | Copernicus COP30 (~30 m horizontal resolution) |
| Terrain elevation range | 1203 m to 1798 m a.s.l. (595 m of relief) |
| Length of navigable lanes in HD map | 15.04 km (truncated to a single route with no forks between two roundabouts to guarantee deterministic runs) |
| Vehicle trajectory per run | ~10 km of winding mountain road |

**Note on map elevation:** the OpenDRIVE network exported from RoadRunner has flat elevation on the lanes (`z=0`), whereas the visual mesh and physical colliders preserve the DEM elevation. As a result, the EV model reads the grade from the **vehicle's instantaneous pitch angle**, not from the waypoint network. This approach was validated against literature predictions for grade-induced energy consumption.

---

## 2. Experimental design

A **2 × 2 × 3 factorial design** with 12 scenarios:

- **Factor 1:** Vehicle (Tesla Model 3 RWD, Audi e-tron 55 quattro)
- **Factor 2:** Initial state of charge (90%, 40%)
- **Factor 3:** Weather (dry/24°C, wet/10°C, snow/-5°C)

Each run consists of a ~10 km traversal of the truncated GU-186 segment, with CARLA's Traffic Manager driving the vehicle autonomously and logging kinematic data at 20 Hz. Energy quantities are computed in Python from the kinematic trace using a custom EV model.

### 2.1 Vehicle parameters

The parameters are stored in `ev_model.py` under the `EV_LIBRARY` dictionary. Sources: manufacturer specifications, EPA test data, ev-database.org, ADAC tests.

| Parameter | Tesla Model 3 RWD | Audi e-tron 55 quattro |
|---|---|---|
| Curb mass (kg) | 1611 | 2490 |
| Drag coefficient $C_d$ | 0.23 | 0.27 |
| Frontal area $A$ (m²) | 2.22 | 2.65 |
| Rolling resistance coeff. $C_{rr}$ | 0.011 | 0.012 |
| Nominal pack voltage (V) | 350 | 396 |
| Usable battery capacity (kWh) | 72.0 | 86.5 |
| Motor efficiency $\eta_{motor}$ | 0.90 | 0.88 |
| Regen efficiency $\eta_{regen}$ | 0.85 | 0.82 |
| Max regen power (kW) | 60 | 220 |
| Max discharge power (kW) | 210 | 300 |
| Pack internal resistance (Ω) | 0.07 | 0.06 |
| Battery thermal mass (J/K) | 180000 | 230000 |
| Battery dissipation (W/K) | 250 | 320 |
| Base HVAC power (W) | 300 | 400 |

### 2.2 Weather parameters

Three weather conditions, each parameterized consistently between CARLA's WeatherParameters and the EV model's weather-coupling functions:

| Weather tag | $T_{amb}$ (°C) | Pavement condition | Precipitation | Wetness | Wind | Sun altitude |
|---|---|---|---|---|---|---|
| `dry24` | 24 | dry | 0% | 0% | 0% | 30° |
| `wet10` | 10 | wet | 60% | 70% | 30% | 10° |
| `snow_neg5` | -5 | snow | 80% | 40% | 50% | 5° |

**Weather effects in the EV model:**

The energy model applies three multipliers depending on the weather:

1. **HVAC auxiliary power** $P_{aux}(T_{amb})$: piecewise linear, 500 W at 24°C → 1500 W at 10°C → 3750 W at -5°C
2. **Battery capacity factor** (Liu et al., 2018 for Li-ion NMC): 1.00 at 24°C → 0.95 at 10°C → 0.86 at -5°C
3. **Effective rolling resistance** (Sandberg, 2011): ×1.0 dry, ×1.2 wet, ×2.0 snow

**Note on visual rendering:** CARLA 0.9.16 does not render snow particles natively. The "snow" scenarios are therefore characterized by their *physical effects* on the EV model (low $T_{amb}$, high $C_{rr}$, reduced battery capacity, high HVAC), not visually. Rain rendering and surface wetness are present visually in the `wet10` scenarios.

### 2.3 Complete scenario catalog

| ID | Vehicle | SOC | Weather | $T_{amb}$ | Pavement | $P_{precip}$ | Wetness | Wind | Sun |
|---|---|---|---|---|---|---|---|---|---|
| S01 | Tesla Model 3 | 90% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S02 | Tesla Model 3 | 90% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S03 | Tesla Model 3 | 90% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |
| S04 | Tesla Model 3 | 40% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S05 | Tesla Model 3 | 40% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S06 | Tesla Model 3 | 40% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |
| S07 | Audi e-tron | 90% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S08 | Audi e-tron | 90% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S09 | Audi e-tron | 90% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |
| S10 | Audi e-tron | 40% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S11 | Audi e-tron | 40% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S12 | Audi e-tron | 40% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |

---

## 3. Dataset description

Each scenario produces a CSV file named `datos_grid_<ID>.csv` with one row per simulation tick (20 Hz). The 12 files share the same 41 columns.

### 3.1 Naming convention

```
datos_grid_S01_tesla_SOC90_dry24.csv
datos_grid_S02_tesla_SOC90_wet10.csv
...
datos_grid_S12_audi_SOC40_snow_neg5.csv
```

Each file is approximately 25,000 rows (the simulated trajectory lasts ~21 minutes at 20 Hz).

### 3.2 Column schema

| Column | Units | Description |
|---|---|---|
| `scenario_id` | string | Full scenario identifier (e.g. `S01_tesla_SOC90_dry24`) |
| `vehicle_key` | string | Key in `EV_LIBRARY` (`tesla_model3`, `audi_etron`) |
| `run_id` | string | 8-character UUID unique per run, useful if a scenario is re-run |
| `vehicle_id` | int | Actor ID in CARLA |
| `timestamp_abs` | s | Absolute simulator time (since simulator start) |
| `time_relative` | s | Time since the start of this run |
| `dt` | s | Tick duration (constant 0.05 s) |
| `road_id`, `lane_id` | int | OpenDRIVE identifiers (informative; the map is z=0) |
| `pos_x`, `pos_y`, `pos_z` | m | Vehicle position in CARLA's world frame. `pos_z` reflects the actual elevation from the DEM mesh |
| `heading` | degrees | Vehicle yaw angle |
| `road_grade_deg` | degrees | Road grade, read from the vehicle pitch (positive = climbing). Observed range: −19° to +20° |
| `road_curvature` | 1/m | Instantaneous curvature, computed from angular velocity |
| `road_friction` | — | Tire friction parameter in CARLA |
| `speed_kmh` | km/h | Vehicle speed |
| `acc_long`, `acc_lat` | m/s² | Acceleration projected onto the vehicle's forward and right vectors |
| `throttle`, `brake`, `steering` | 0..1 / -1..1 | Autopilot control inputs |
| `gear` | int | Engaged gear |
| `soc_percent` | % | Instantaneous SOC |
| `capacity_usable_wh` | Wh | Effective usable capacity (depends on $T_{amb}$) |
| `voltage_v` | V | Instantaneous pack voltage |
| `current_a` | A | Instantaneous pack current (positive = discharge, negative = charge) |
| `power_watts` | W | Net power at the battery (positive = discharge, negative = regen) |
| `energy_used_cum_wh` | Wh | Cumulative energy consumed since the start of the run |
| `energy_regen_cum_wh` | Wh | Cumulative energy regenerated since the start of the run |
| `is_regen` | 0/1 | 1 if power is negative this tick (regenerating) |
| `battery_temp_c` | °C | Modeled battery temperature |
| `aux_power_w` | W | Instantaneous HVAC/auxiliary power applied |
| `c_rr_eff` | — | Effective rolling resistance coefficient |
| `ambient_temp_c` | °C | Scenario ambient temperature (constant per run) |
| `precip_type` | string | `none` or `rain` |
| `precip_intensity` | 0..100 | Scenario precipitation parameter |
| `wind_speed_ms` | m/s | Scenario wind speed |
| `wind_dir_deg` | degrees | Scenario wind direction |
| `wind_long_ms` | m/s | Longitudinal wind component (positive = headwind) |
| `road_condition` | string | `dry`, `wet`, or `snow` |

### 3.3 Run termination

A run ends when one of the following conditions is met:

1. **Distance target reached:** the vehicle has accumulated ≥ 10,000 m of physical travel (the truncated route is ~10.5 km end to end).
2. **Stuck detector:** the vehicle has moved less than 10 m within a 30 s window (i.e. it is stuck at a roundabout or obstacle). In this case `stop_reason = stuck`.
3. **Safety timeout:** 3600 s of simulated time (60 min). Never triggered in this dataset.

All 12 runs in this dataset terminated normally by condition 1 (`reached_destination`).

---

## 4. Master results table

All runs reached the 10 km target in ~1250 s of simulated time, traveling between elevations of 1242 m (valley) and ~1460 m (upper roundabout area), with a cumulative ascent of ~412 m and descent of ~254 m. The route is winding, which provides a broad distribution of grade bins per run (from −19° to +20°).

| # | Vehicle | SOC init | Weather | SOC final | SOC drop | E used | E regen | E net | Wh/km |
|---|---|---|---|---|---|---|---|---|---|
| S01 | Tesla | 90% | dry 24°C | 87.65% | 2.35 pts | 2472 Wh | 781 Wh | 1691 Wh | 169.1 |
| S02 | Tesla | 90% | wet 10°C | 86.93% | 3.07 pts | 2761 Wh | 663 Wh | 2099 Wh | 209.9 |
| S03 | Tesla | 90% | snow -5°C | 84.58% | 5.42 pts | 3781 Wh | 418 Wh | 3363 Wh | 336.3 |
| S04 | Tesla | 40% | dry 24°C | 37.65% | 2.35 pts | 2472 Wh | 781 Wh | 1691 Wh | 169.1 |
| S05 | Tesla | 40% | wet 10°C | 36.76% | 3.24 pts | 2880 Wh | 663 Wh | 2217 Wh | 221.7 |
| S06 | Tesla | 40% | snow -5°C | 34.58% | 5.42 pts | 3781 Wh | 418 Wh | 3363 Wh | 336.3 |
| S07 | Audi | 90% | dry 24°C | 86.73% | 3.27 pts | 3897 Wh | 1068 Wh | 2829 Wh | 282.9 |
| S08 | Audi | 90% | wet 10°C | 86.00% | 4.00 pts | 4210 Wh | 924 Wh | 3286 Wh | 328.6 |
| S09 | Audi | 90% | snow -5°C | 83.77% | 6.23 pts | 5269 Wh | 623 Wh | 4646 Wh | 464.6 |
| S10 | Audi | 40% | dry 24°C | 36.73% | 3.27 pts | 3897 Wh | 1068 Wh | 2829 Wh | 282.9 |
| S11 | Audi | 40% | wet 10°C | 36.10% | 3.90 pts | 4122 Wh | 915 Wh | 3207 Wh | 320.7 |
| S12 | Audi | 40% | snow -5°C | 33.72% | 6.28 pts | 5296 Wh | 612 Wh | 4684 Wh | 468.4 |

### 4.1 Net energy matrix (Wh per 10 km traversal)

|  | dry 24°C | wet 10°C | snow -5°C |
|---|---|---|---|
| **Tesla SOC 90%** | 1691 | 2099 | 3363 |
| **Tesla SOC 40%** | 1691 | 2217 | 3363 |
| **Audi SOC 90%**  | 2829 | 3286 | 4646 |
| **Audi SOC 40%**  | 2829 | 3207 | 4684 |

### 4.2 SOC drop matrix (percentage points per 10 km)

|  | dry 24°C | wet 10°C | snow -5°C |
|---|---|---|---|
| **Tesla SOC 90%** | 2.35 | 3.07 | 5.42 |
| **Tesla SOC 40%** | 2.35 | 3.24 | 5.42 |
| **Audi SOC 90%**  | 3.27 | 4.00 | 6.23 |
| **Audi SOC 40%**  | 3.27 | 3.90 | 6.28 |

---

## 5. Key findings

### 5.1 Weather has a larger effect than vehicle choice

Net energy in snow can be roughly **double** that of dry conditions:

- Tesla: snow consumes **+99% more** than dry (1691 → 3363 Wh)
- Audi: snow consumes **+64% more** than dry (2829 → 4646 Wh)

The Tesla is *more* sensitive to weather in relative terms because its baseline is lower, so the same HVAC and rolling-resistance penalties weigh more heavily on its proportionally smaller consumption. This suggests that **efficient EVs experience proportionally larger weather-induced penalties**, an insight that may be of interest when extrapolating fleet-level range planning to mountain corridors under adverse conditions.

### 5.2 Vehicle dominance narrows with weather severity

The Audi consumes much more than the Tesla in dry conditions, but the gap narrows in snow:

| Weather | Audi / Tesla ratio | Gap |
|---|---|---|
| dry | 1.67× | +67.3% |
| wet | 1.57× | +56.6% |
| snow | 1.38× | +38.1% |

This happens because the weather-induced load (HVAC, $C_{rr}$, capacity reduction) is largely vehicle-independent in absolute terms, so it appears as a smaller relative penalty on the Audi's already higher consumption. The e-tron's peak regen power (220 kW vs 60 kW for the Model 3) also contributes to closing the gap in scenarios with significant descents.

### 5.3 Initial SOC barely affects net consumption

A central finding for the physical validation of the model: under equivalent conditions (same vehicle, same weather), the SOC at which the trajectory begins does **not** alter the net energy consumed by the trajectory. This is observed in:

- S04 ≡ S01 (Tesla dry): both 1691 Wh
- S06 ≡ S03 (Tesla snow): both 3363 Wh
- S10 ≡ S07 (Audi dry): both 2829 Wh
- S12 ≈ S09 (Audi snow): 4684 vs 4646 Wh (Δ = +38 Wh, +0.8%)

Only in `wet` conditions does a small effect emerge, with **opposite signs** between vehicles:

- S05 vs S02 (Tesla wet): +119 Wh (+5.65%) when starting at SOC 40%
- S11 vs S08 (Audi wet, mean of two re-runs): −105 Wh (−3.2%) when starting at SOC 40%

This asymmetry between vehicles is explained by the interaction of:
- $I^2 R$ losses (favor SOC 90% because higher pack voltage means lower current for the same power)
- Regenerative-braking saturation limits (favor SOC 40% on vehicles with high regen capacity, because lower voltage allows absorbing more mechanical power before saturating `max_regen_power`)

For the Tesla (60 kW regen ceiling), the $I^2 R$ effect dominates → *higher* net consumption at SOC 40%.
For the Audi (220 kW regen ceiling), the saturation-relief effect dominates → *lower* net consumption at SOC 40%.

This is a non-trivial empirical observation that motivates richer, powertrain-aware modeling in future work.

### 5.4 Range-anxiety quantification (SOC 40% scenarios)

Projecting the observed SOC drop rate over 10 km of mountain road to longer trajectories, assuming that weather conditions and grade distribution remain similar:

| Scenario | SOC drop over 10 km | projected km until SOC = 10% (mountain conditions) |
|---|---|---|
| Tesla SOC 40% dry | 2.35 pts | 128 km |
| Tesla SOC 40% wet | 3.24 pts | 83 km |
| Tesla SOC 40% snow | 5.42 pts | **55 km** |
| Audi SOC 40% dry | 3.27 pts | 82 km |
| Audi SOC 40% wet | 3.90 pts | 67 km |
| Audi SOC 40% snow | 6.28 pts | **48 km** |

The two worst scenarios (S06 and S12) imply that a driver who started a 10 km mountain pass at SOC 40% in snow could continue **an additional 38–45 km** before reaching the conventional 10% safety reserve. This is a much shorter range than a naive estimate based on highway consumption would suggest, motivating the need for terrain- and weather-aware energy-warning systems (such as the V2X-based system the broader project aims to develop).

---

## 6. Reproducibility

CARLA's Traffic Manager introduces a small amount of stochasticity into throttle and steering control across runs. To assess this:

- **S11 was run twice** (run 1 and run 2) under identical scenario parameters.
- Net energy: 3207 Wh (run 1) vs 3156 Wh (run 2)
- Variability: ±25 Wh over a mean of 3181 Wh, **CV ≈ 0.8%**

This confirms that the scenario-to-scenario differences reported in this dataset (typically tens to hundreds of Wh) are dominated by the controlled experimental factors, not by autopilot noise.

---

## 7. Repository contents

```
.
├── README.md                              # this document
├── ev_model.py                            # EV energy model with vehicle library
├── seccionA_B_C_D_E_v5_2.py               # CARLA acquisition script with the 12-scenario catalog
├── grid_runs/
│   ├── datos_grid_S01_tesla_SOC90_dry24.csv
│   ├── datos_grid_S02_tesla_SOC90_wet10.csv
│   ├── datos_grid_S03_tesla_SOC90_snow_neg5.csv
│   ├── datos_grid_S04_tesla_SOC40_dry24.csv
│   ├── datos_grid_S05_tesla_SOC40_wet10.csv
│   ├── datos_grid_S06_tesla_SOC40_snow_neg5.csv
│   ├── datos_grid_S07_audi_SOC90_dry24.csv
│   ├── datos_grid_S08_audi_SOC90_wet10.csv
│   ├── datos_grid_S09_audi_SOC90_snow_neg5.csv
│   ├── datos_grid_S10_audi_SOC40_dry24.csv
│   ├── datos_grid_S11_audi_SOC40_wet10.csv
│   └── datos_grid_S12_audi_SOC40_snow_neg5.csv
└── 
```

---

## 8. How to read a CSV in Python

```python
import pandas as pd

df = pd.read_csv('grid_runs/datos_grid_S03_tesla_SOC90_snow_neg5.csv')

print(f"Scenario: {df['scenario_id'].iloc[0]}")
print(f"Vehicle:  {df['vehicle_key'].iloc[0]}")
print(f"Duration: {df['time_relative'].iloc[-1]:.1f} s")
print(f"SOC: {df['soc_percent'].iloc[0]:.2f}% -> {df['soc_percent'].iloc[-1]:.2f}%")
print(f"Net energy: {df['energy_used_cum_wh'].iloc[-1] - df['energy_regen_cum_wh'].iloc[-1]:.0f} Wh")

# Energy consumption stratified by road grade
import numpy as np
bins = [-20, -10, -5, -1, 1, 5, 10, 20]
df['grade_bin'] = pd.cut(df['road_grade_deg'], bins=bins)
df['e_step'] = df['power_watts'] * df['dt'] / 3600.0   # Wh per tick
# (compute distance traveled per tick separately to obtain Wh/km per bin)
```

---

## 9. Acknowledgments and references

**Geospatial data:** Copernicus COP30 DEM (European Space Agency); road geometry from OpenStreetMap.

**Model calibration sources:**
- Liu, K., et al. (2018). *A brief review on key technologies in the battery management system of electric vehicles*. Frontiers of Mechanical Engineering. (Battery capacity vs temperature)
- Sandberg, U. (2011). *Rolling resistance — basic information and state-of-the-art on measurement methods*. (Rolling resistance multipliers under wet/snow conditions)

**Software:** CARLA Sim 0.9.16 (Unreal Engine 4), MathWorks RoadRunner R2024a, MATLAB R2024a, Python 3.12 with NumPy, Pandas.

---

## 10. Contact

Repository maintained by Emmanuel Cuevas, with collaborators Yeray Mezquita Martín, Albano Carrera Gonzáles, and Diego Valdeolmillos Villaverde. For questions about the data, model assumptions, or experimental design, please open an issue in this repository.
