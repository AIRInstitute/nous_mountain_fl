# Mountain-Pass EV Energy: Digital-Twin Dataset and Federated Beacon Experiments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21159398.svg)](https://doi.org/10.5281/zenodo.21159398)
[![License: MIT](https://img.shields.io/badge/License%3A%20code-MIT-blue.svg)](LICENSE)
[![License: CC0-1.0](https://img.shields.io/badge/License%3A%20data-CC0--1.0-lightgrey.svg)](LICENSE-DATA)

This repository is the public research artifact accompanying a manuscript in preparation for
*Open Research Europe (ORE)*. It brings together, in a single place, everything needed to
reproduce the reported results:

1. a **digital-twin dataset** of electric-vehicle (EV) traversals of a representative
   mountain-road segment — the **Puerto de la Quesera** (Sistema Central, GU-186 road), and
2. the **federated-learning (FL) experiments** that turn that dataset into a *mountain-pass
   beacon*: a local, connectivity-free V2I server that estimates, on entry, the energy a
   vehicle needs to cross the pass and verifies it, on exit, against the energy actually
   consumed.

The scientific motivation is terrain- and weather-aware range planning on mountain corridors,
where a naive highway-consumption estimate can badly overstate the remaining range.

---

## Repository layout

```
nous_mountain_fl/
├── README.md                  # this overview
├── CITATION.cff               # how to cite this artifact
├── LICENSE                    # MIT — applies to ALL software in this repository
├── LICENSE-DATA               # CC0-1.0 — applies to the datasets in data/
├── data/                      # ── DATA (CC0-1.0) ──
│   ├── README.md              # full dataset documentation (schema, catalog, results)
│   └── datos_grid_S01..S12.csv    # 12 EV traversal logs (2 vehicles × 2 SOC × 3 weather)
├── digital_twin/              # ── CODE: data generation (MIT) ──
│   ├── README.md
│   ├── ev_model.py            # longitudinal EV energy model + vehicle library
│   └── carla_acquisition.py   # CARLA acquisition script with the 12-scenario catalog
├── federated_experiments/     # ── CODE + CONFIGURATION: the experiments (MIT) ──
│   ├── README.md
│   ├── pyproject.toml  requirements.txt
│   ├── configs/               # cross-validation and prequential-beacon configurations
│   ├── scripts/               # POSIX + Windows launchers
│   └── src/mountain_pass_fl/  # preprocessing, baselines, MLP, Flower FL, prequential eval
├── preprocess/                 # ── SOFTWARE: DEM-to-RoadRunner HD Map middleware (MIT) ──
│   ├── README.md
│   ├── LICENSE                 # MIT
│   ├── CITATION.cff
│   ├── mapperV3.mlapp          # MATLAB App Designer source
│   └── mapperInstaller_web.exe # standalone Windows installer (no MATLAB license required)
└── carla_mountain/              # ── SOFTWARE (MIT) + MAP DATA (CC-BY-4.0) — see License below ──
    ├── README.md
    ├── SETUP.md
    ├── CITATION.cff
    ├── LICENSE-code             # MIT — example scripts
    ├── LICENSE-data             # CC-BY-4.0 — map assets (OSM/Copernicus attribution required)
    └── examples/                # 01_check_elevation.py
```
> The `Mountain_0.9.16.zip` CARLA import bundle (288 MB) is **not** stored in this repository —
> it's distributed via the [Digital-Twin ORE Google Drive folder](https://drive.google.com/drive/folders/1-sQKS117Wj-J6C-D6ptcXz5CtvUCtwj-?usp=sharing),
> linked from [`carla_mountain/README.md`](carla_mountain/README.md).

### How the layout maps to the paper's *Data and Software Availability*

| Paper section | Location |
|---|---|
| **Data** — 12 simulated EV traversal logs | [`data/`](data/) |
| **Code** — preprocessing, baselines, residual-model training, Flower FL, prequential beacon evaluation | [`digital_twin/`](digital_twin/) and [`federated_experiments/`](federated_experiments/) |
| **Configuration** — cross-validation and prequential-beacon settings | [`federated_experiments/configs/`](federated_experiments/configs/) |
| **Software** — DEM-to-RoadRunner HD Map middleware | [`preprocess/`](preprocess/) |
| **Software + map data** — CARLA 0.9.16 digital twin of the pass | [`carla_mountain/`](carla_mountain/) |
---

## The five components

### 1. Data — [`data/`](data/README.md)
Twelve CSV logs from a **2 × 2 × 3 factorial design** (Tesla Model 3 RWD / Audi e-tron 55
quattro × initial SOC 90 % / 40 % × dry-24 °C / wet-10 °C / snow-−5 °C). Each file holds one
row per simulation tick (20 Hz, ~25 000 rows) with time-stamped kinematic, positional, weather
and battery-level variables. See the [dataset README](data/README.md) for the full column
schema, scenario catalog, master results tables, and key findings.

### 2. Digital twin — [`digital_twin/`](digital_twin/README.md)
The generator behind the dataset: the **DEM → MATLAB → RoadRunner → CARLA 0.9.16 → Python**
pipeline, the longitudinal energy model with weather coupling (`ev_model.py`), and the CARLA
acquisition script (`carla_acquisition.py`).

### 3. Federated experiments — [`federated_experiments/`](federated_experiments/README.md)
The FL proof of concept. A residual MLP learns a pass-specific correction on top of a flat
physical baseline, and is trained centrally and with Flower (FedAvg / FedProx). A prequential
"predict-on-entry, update-on-exit" protocol evaluates the beacon as cars arrive. Includes
route-level cross-validation and a progressive beacon learning curve.

### 4. Geospatial preprocessing — [`preprocess/`](preprocess/README.md)
The MATLAB App Designer tool (`mapperV3`) that produced the terrain-aware RoadRunner HD Map
used to build the CARLA map below. Converts a GeoTIFF DEM and an OpenStreetMap road Shapefile
into a `.rrhd` HD Map with real elevation fused into the road geometry itself — not draped as a
flat mesh underneath it. Ships with a standalone Windows executable, no MATLAB license needed.

### 5. CARLA map — [`carla_mountain/`](carla_mountain/README.md)
The finished CARLA 0.9.16 digital twin of the Puerto de la Quesera (33 roads, 884 elevation
primitives, 1242–1494 m a.s.l.), packaged as an import bundle for any precompiled CARLA
installation. Includes the road-grade caveat (read pitch from the vehicle transform, not from
waypoints) and a diagnostic script that reproduces it.

---

## Quick start (federated experiments)

The FL experiments are the reproducible core. From [`federated_experiments/`](federated_experiments/):

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .

# Extended run: cross-validation + beacon learning curve (reads ../data)
.venv/bin/python -m mountain_pass_fl.run_extended \
  --data-dir ../data --out-dir outputs_extended \
  --config configs/extended.yaml --rebuild-segments
```

Windows PowerShell/`.bat` launchers are provided in
[`federated_experiments/scripts/`](federated_experiments/scripts/). Full instructions,
outputs, and metrics are documented in the
[federated experiments README](federated_experiments/README.md).

Federated experiments use `flwr[simulation]` in the 1.x API range (`>=1.8,<2.0`); the core
stack is NumPy, pandas, scikit-learn, PyTorch, Flower, Matplotlib and PyYAML.

To instead build or modify the digital twin itself, start with
[`preprocess/README.md`](preprocess/README.md) (MATLAB tool) and
[`carla_mountain/README.md`](carla_mountain/README.md) (CARLA map).

---

## Citation

If you use this dataset or code, please cite the archived artifact (and, once available, the
accompanying manuscript). Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

- **Concept DOI (always resolves to the latest version):** [`10.5281/zenodo.21159398`](https://doi.org/10.5281/zenodo.21159398)

Cite the concept DOI unless you need to point to a specific release.

## License

This artifact is **dual-licensed**: all *software* is released under the OSI-approved
[MIT License](https://opensource.org/license/mit), and all *data* under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (public domain
dedication). The only exception is the CARLA map package, whose geometry derives from
OpenStreetMap and Copernicus and therefore carries an upstream attribution requirement.

| Component | Path | License | File |
|---|---|---|---|
| Federated-learning experiments, preprocessing, baselines, models | `federated_experiments/` | **MIT** | [`LICENSE`](LICENSE) |
| EV energy model and CARLA acquisition scripts | `digital_twin/` | **MIT** | [`LICENSE`](LICENSE) |
| DEM-to-RoadRunner HD Map middleware (`mapperV3`) | `preprocess/` | **MIT** | [`preprocess/LICENSE`](preprocess/LICENSE) |
| CARLA example scripts | `carla_mountain/examples/` | **MIT** | [`carla_mountain/LICENSE-code`](carla_mountain/LICENSE-code) |
| 12 simulated EV traversal logs + derived result tables | `data/` | **CC0-1.0** | [`LICENSE-DATA`](LICENSE-DATA) |
| CARLA map assets (`Mountain.fbx`, `.xodr`, `.json`, import bundle) | `carla_mountain/` | **CC-BY-4.0** | [`carla_mountain/LICENSE-data`](carla_mountain/LICENSE-data) |

The map assets stay under CC-BY-4.0 — not CC0 — because their road network derives from
**OpenStreetMap** ([ODbL 1.0](https://www.openstreetmap.org/copyright)) and their elevation from
the **Copernicus GLO-30 DEM**, both of which require attribution downstream. The 12 traversal
logs, by contrast, are entirely original simulation output and carry no upstream obligation, so
they are dedicated to the public domain under CC0.

Software availability, in the form requested by *Open Research Europe*:

- **Source code available from:** https://github.com/AIRInstitute/nous_mountain
- **Archived software available from:** https://doi.org/10.5281/zenodo.21159398
- **License:** [MIT](https://opensource.org/license/mit)

## Acknowledgments and references

This research has been supported by the project “A catalyst for EuropeaN ClOUd Services in the era of data spaces, high-performance and edge computing (NOUS)”, Grant Agreement Number 101135927. Funded by the European Union, views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

- **Geospatial data:** Copernicus COP30 DEM (European Space Agency); road geometry from
  OpenStreetMap.
- **Model-calibration sources:** Liu, K., et al. (2018), *Frontiers of Mechanical Engineering*
  (battery capacity vs temperature); Sandberg, U. (2011), rolling-resistance multipliers under
  wet/snow conditions.
- **Software:** CARLA Sim 0.9.16 (Unreal Engine 4), MathWorks RoadRunner R2024a, MATLAB
  R2024a, Python 3.12.

## Contact

Developed by Jesus Emmanuel Vidal Cuevas, Yeray Mezquita Martín, Albano Carrera González, and Diego
Valdeolmillos Villaverde. For questions about the data, model assumptions, or experimental
design, please open an issue.
