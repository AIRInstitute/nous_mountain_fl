# SETUP — installing CARLA, importing `Mountain`, and setting up the Python client

This guide assumes you want to **consume** the `Mountain` map, i.e. you're using the precompiled CARLA 0.9.16 release, not building from source. For source builds see [`README.md`](README.md#6-building-from-source-instead).

## 1. Install precompiled CARLA 0.9.16

1. Download the CARLA 0.9.16 precompiled package for your OS from the [official CARLA releases page](https://github.com/carla-simulator/carla/releases).
2. Extract it to a directory with no spaces in the path (Windows users especially — CARLA's batch scripts are picky about this).
3. Confirm the base maps load before adding `Mountain`:
   ```bash
   ./CarlaUE4.sh          # Linux
   CarlaUE4.exe           # Windows
   ```

## 2. Import the `Mountain` map

1. Download `Mountain_0.9.16.tar.gz` from the [Digital-Twin ORE](https://drive.google.com/drive/folders/1LiuKO9zR1adKe38_wBr-CXa-3FNJfNlo?usp=drive_link) Google Drive folder (it's 288 MB — too large to be stored in the git tree itself).
2. Place it in `Import/` inside your CARLA installation directory.
3. Run CARLA's own import script from the installation root:
   ```bash
   ./ImportAssets.sh      # Linux
   ImportAssets.bat       # Windows
   ```
4. This unpacks the map and registers it with CARLA's asset system. No manual UE4 editing is required — that's the whole point of using the precompiled path.

## 3. Set up the Python client environment (Anaconda)

The experimental campaign on this map used an Anaconda environment named `carla_env` with Python 3.12.

```bash
conda create -n carla_env python=3.12 -y
conda activate carla_env
pip install carla==0.9.16
```

If the exact wheel for your platform isn't on PyPI, use the `.whl` shipped inside the CARLA installation's `PythonAPI/carla/dist/` folder instead:

```bash
pip install /path/to/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-<platform-tag>.whl
```

## 4. Load the map and run a smoke test

1. Start the CARLA server:
   ```bash
   ./CarlaUE4.sh -quality-level=Epic
   ```
2. In a second terminal (with `carla_env` active), load the map:
   ```bash
   python config.py -m Mountain --delta-seconds 0.05
   ```
3. Run the elevation diagnostic from this repo to confirm the map imported correctly and to see the road-grade caveat from [`README.md`](README.md#4-elevation-caveat--read-this-before-you-use-waypoints) in action:
   ```bash
   python examples/01_check_elevation.py
   ```
   Expected output: confirmation that the `.xodr` waypoint elevation is z = 0, and that vehicle-pitch-based grade is non-zero and varies along the route.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportAssets` finishes but `Mountain` doesn't appear in `config.py --list` | Bundle placed outside `Import/` | Re-check the path; it must sit directly in `<carla_root>/Import/`. |
| Server crashes on map load with a shader/texture error | GPU driver too old for UE4 4.26 | Update your GPU driver; CARLA 0.9.16 requires a reasonably recent one. |
| Python client connects but times out spawning a vehicle | Map still loading (large terrain mesh) | Wait a few extra seconds after `config.py -m Mountain` before spawning actors. |
| `waypoint.transform.location.z` is always 0 | Expected — see [README.md, S4](README.md#4-elevation-caveat--read-this-before-you-use-waypoints) | Use `vehicle.get_transform().rotation.pitch` for road grade instead. |
| `pip install carla==0.9.16` fails to find a wheel | No prebuilt wheel for your Python/OS combo on PyPI | Use the `.whl` bundled with your CARLA installation (see S3). |

## 6. Where to go next

- To understand *how* the map was built (DEM + OSM fusion), see [`../preprocess`](../preprocess/README.md).
- To reproduce the EV energy experiments run on this map, see the manuscript in preparation and the `examples/` scripts in this repo.
