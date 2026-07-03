# Mountain-Pass EV Energy: Digital-Twin Dataset and Federated Beacon Experiments

[![DOI](https://zenodo.org/badge/1110671604.svg)](https://doi.org/10.5281/zenodo.21159398)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

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
├── data/                      # ── DATA ──
│   ├── README.md              # full dataset documentation (schema, catalog, results)
│   └── datos_grid_S01..S12.csv    # 12 EV traversal logs (2 vehicles × 2 SOC × 3 weather)
├── digital_twin/              # ── CODE: data generation ──
│   ├── README.md
│   ├── ev_model.py            # longitudinal EV energy model + vehicle library
│   └── carla_acquisition.py   # CARLA acquisition script with the 12-scenario catalog
└── federated_experiments/     # ── CODE + CONFIGURATION: the experiments ──
    ├── README.md
    ├── pyproject.toml  requirements.txt
    ├── configs/               # cross-validation and prequential-beacon configurations
    ├── scripts/               # POSIX + Windows launchers
    └── src/mountain_pass_fl/  # preprocessing, baselines, MLP, Flower FL, prequential eval
```

### How the layout maps to the paper's *Data and Software Availability*

| Paper section | Location |
|---|---|
| **Data** — 12 simulated EV traversal logs | [`data/`](data/) |
| **Code** — preprocessing, baselines, residual-model training, Flower FL, prequential beacon evaluation | [`digital_twin/`](digital_twin/) and [`federated_experiments/`](federated_experiments/) |
| **Configuration** — cross-validation and prequential-beacon settings | [`federated_experiments/configs/`](federated_experiments/configs/) |

---

## The three parts

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

---

## Citation

If you use this dataset or code, please cite the archived artifact (and, once available, the
accompanying manuscript). Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

- **Concept DOI (always resolves to the latest version):** [`10.5281/zenodo.21159398`](https://doi.org/10.5281/zenodo.21159398)
- **This version (v1.0.0):** [`10.5281/zenodo.21159399`](https://doi.org/10.5281/zenodo.21159399)

Cite the concept DOI unless you need to point to a specific release.

## License

This work is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**
license — see [`LICENSE`](LICENSE). You are free to share and adapt the material for any
purpose, provided you give appropriate credit.

## Acknowledgments and references

- **Geospatial data:** Copernicus COP30 DEM (European Space Agency); road geometry from
  OpenStreetMap.
- **Model-calibration sources:** Liu, K., et al. (2018), *Frontiers of Mechanical Engineering*
  (battery capacity vs temperature); Sandberg, U. (2011), rolling-resistance multipliers under
  wet/snow conditions.
- **Software:** CARLA Sim 0.9.16 (Unreal Engine 4), MathWorks RoadRunner R2024a, MATLAB
  R2024a, Python 3.12.

## Contact

Developed by Emmanuel Cuevas, Yeray Mezquita Martín, Albano Carrera González, and Diego
Valdeolmillos Villaverde. For questions about the data, model assumptions, or experimental
design, please open an issue.
