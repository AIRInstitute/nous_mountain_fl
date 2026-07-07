# Puerto de la Quesera digital twin for CARLA 0.9.16

**A high-fidelity digital twin of a real Spanish mountain pass, packaged as an import bundle for CARLA 0.9.16. Drop it into a precompiled CARLA installation and drive a real mountain road with genuine terrain elevation.**

This map — internally named `Mountain` — is the CARLA-side artefact of a digital-twin pipeline that starts in [`../preprocess`](../preprocess/README.md) (the MATLAB App Designer tool that fuses a DEM and OpenStreetMap roads into a terrain-aware RoadRunner HD Map). This folder picks up from the finalized RoadRunner scene and documents everything needed to get it running inside CARLA.

A scientific manuscript describing the full methodology and the EV energy experiments run on this map is in preparation for *Open Research Europe*.

---

## Table of contents

1. [What this repository contains](#1-what-this-repository-contains)
2. [The case study](#2-the-case-study)
3. [How the map was produced](#3-how-the-map-was-produced)
4. [Elevation caveat — read this before you use waypoints](#4-elevation-caveat--read-this-before-you-use-waypoints)
5. [Quick start](#5-quick-start)
6. [Building from source instead](#6-building-from-source-instead)
7. [Repository layout](#7-repository-layout)
8. [Citing this package](#8-citing-this-package)
9. [License](#9-license)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. What this repository contains

This repo contains everything needed to **add the `Mountain` map to an existing precompiled CARLA 0.9.16 installation** (the kind most users download from CARLA's official release page). It does **not** distribute CARLA itself.

| File | Purpose |
|---|---|
| `Mountain_0.9.16.zip` | Import bundle for CARLA. **Not stored in this folder** — it's 288 MB, over GitHub's plain-git file limit. Contains the cooked UE4 map, the `.xodr` network, and the metadata `ImportAssets` expects. |
| `source/Mountain.fbx` | Raw mesh exported from RoadRunner R2024a. Use only if you want to re-import the scene into UE4 from source. |
| `source/Mountain.xodr` | Raw OpenDRIVE 1.4 network with 884 elevation primitives between 1242 m and 1494 m. Same notes as the FBX. |
| `source/Mountain.fbm` | RoadRunner scene assets file. |
| `examples/` | Minimal Python client scripts: connect, spawn a vehicle, drive on autopilot, dump a CSV of pose + grade. |
| `SETUP.md` | Full installation and environment guide (CARLA + Anaconda). |

## 2. The case study

The map is a digital twin of a winding ~10 km fragment of the **GU-186 road / Puerto de la Quesera** mountain pass (Sistema Central, on the border between Guadalajara and Segovia provinces, Spain), built from a Copernicus GLO-30 DEM and OpenStreetMap road vectors.

- **Roads:** 33
- **Elevation primitives:** 884 `<elevation>` entries in the `.xodr`
- **Elevation band actually traversed by the road:** 1242.4–1493.5 m a.s.l.
- **Wider terrain relief covered by the DEM tile:** 1203–1798 m a.s.l.
- **UE4 version:** 4.26.2

## 3. How the map was produced

1. **Geospatial preprocessing** in [`../preprocess`](../preprocess/README.md): a DEM and an OSM road Shapefile are fused into a terrain-aware RoadRunner HD Map (`.rrhd`).
2. **Scene finalization + CARLA export** in RoadRunner R2024a, via *File → Export → CARLA Filmbox*, with:
   - Filmbox options: Split by Segmentation, Embed Textures, Export Only Highest LODs, Apply Material Color To Texture.
   - OpenDRIVE options: OpenDRIVE 1.4, right-hand driving, Enforce connected road continuity, Export signals, Export objects, Export OpenCRG with Road Data Format = LRFI.
   
   This combination — found after iterative testing — is what preserves the elevation profile of every road in the exported `.xodr`.
3. **UE4 import**, from a source-built CARLA workspace: the `.fbx + .xodr + .json + .fbm` set is dropped into `Unreal/CarlaUE4/Content/Carla/ExportedMaps/` and registered via `make import`.
4. **Packaging**: `make package ARGS="--packages=Mountain"` produces the deployable `Mountain_0.9.16.zip` bundle, distributed via the [Digital-Twin ORE](https://drive.google.com/drive/folders/1LiuKO9zR1adKe38_wBr-CXa-3FNJfNlo?usp=drive_link) Google Drive folder (see S1 — it's too large to live in the git tree itself).

## 4. Elevation caveat — read this before you use waypoints

The exported HD Map preserves elevation in the visual mesh and physics colliders, in the lane geometry of the `.rrhd`, and in the `<elevationProfile>` blocks of the `.xodr`. **However**, the elevation field of the navigable waypoint network served by CARLA's Python API (`waypoint.transform.location.z`) returns **z = 0** in this configuration — this is a widely reported CARLA/OpenDRIVE limitation, not a bug specific to this map.

**To get a physically consistent road-grade signal, read it from the vehicle's own transform instead of from waypoints:**

```python
pitch_deg = vehicle.get_transform().rotation.pitch  # already normalised to [-90°, +90°]
```

This pitch-based grade signal was validated against literature-based predictions of grade-induced energy consumption on equivalent slopes, and is the method used throughout the experimental campaign run on this map. See `examples/01_check_elevation.py` for a runnable diagnostic that reproduces this finding.

## 5. Quick start

1. Have a precompiled CARLA 0.9.16 installation ready (see [`SETUP.md`](SETUP.md) if you don't).
2. Download `Mountain_0.9.16.zip` from the [Digital-Twin ORE](https://drive.google.com/drive/folders/1LiuKO9zR1adKe38_wBr-CXa-3FNJfNlo?usp=drive_link) Google Drive folder (it's not stored in the git tree — see S1).
3. Import it with CARLA's own import script:
   ```bash
   ./ImportAssets.sh   # or ImportAssets.bat on Windows
   ```
4. Launch the CARLA server, then load the map:
   ```bash
   python config.py -m Mountain --delta-seconds 0.05
   ```
5. Run one of the example scripts in `examples/` to spawn a vehicle and confirm everything works.

Full step-by-step instructions, including the Anaconda Python client environment, are in [`SETUP.md`](SETUP.md)

## 6. Building from source instead

If you need to modify the map (add roads, adjust terrain), you'll need a **source build** of CARLA, not the precompiled one:

```bash
# from a CARLA source workspace, with source/Mountain.fbx + .xodr + .rrdata.xml
# copied into Unreal/CarlaUE4/Content/Carla/ExportedMaps/
make import
make package ARGS="--packages=Mountain"
```

A source build is heavy (≥ 100 GB disk, ~4 h build time on a modern workstation). If you only need to *consume* the map, use the precompiled bundle instead — it's much simpler. Original geospatial preprocessing settings are documented in [`../preprocess/README.md` S6](../preprocess/README.md#6-from-rrhd-to-a-carla-package).

## 7. Repository layout

```
carla_mountain/
├── source/
│   ├── Mountain.fbx
│   ├── Mountain.xodr
│   ├── Mountain.json
│   └── Mountain.fbm
├── examples/
│   └── 01_check_elevation.py
├── README.md
├── SETUP.md
└── CITATION.cff
```

`Mountain_0.9.16.zip` (288 MB) is **not** part of this tree — it's distributed via the [Digital-Twin ORE](https://drive.google.com/drive/folders/1LiuKO9zR1adKe38_wBr-CXa-3FNJfNlo?usp=drive_link) Google Drive folder, since it exceeds the plain-git file size limit.

## 8. Citing this package

If you use the `Mountain` map in academic work, please cite both this package and the upstream paper:

> Vidal Cuevas, J. E., Mezquita Martín, Y., Valdeolmillos, D., & Carrera González, A. (2026). *nous_mountain_carla: a CARLA 0.9.16 digital twin of the Puerto de la Quesera mountain pass* [Software]. GitHub. https://github.com/AIRInstitute/nous_mountain_fl/tree/main/carla

> Vidal Cuevas, J. E., Mezquita Martín, Y., Valdeolmillos, D., & Carrera González, A. (2026). *Beacon-assisted range-risk estimation for electric vehicles in low-coverage mountain roads: a digital-twin study on the Puerto de la Quesera* [Manuscript in preparation].

A `CITATION.cff` file is included for GitHub's automatic citation feature.

## 9. License

| Asset | License | Rationale |
|---|---|---|
| Example Python scripts in `examples/` | [MIT](LICENSE-code) | Matches the upstream `../preprocess` folder. |
| `Mountain.fbm`, `Mountain.fbx`, `Mountain.xodr`, `Mountain.json`, `Mountain_0.9.16.zip` | [CC-BY-4.0](LICENSE-data) | Data-style license; requires attribution to the authors and upstream open-data providers (Copernicus, OpenStreetMap). |
| Documentation (`README*.md`, `SETUP*.md`) | [CC-BY-4.0](LICENSE-data) | Free reuse with attribution. |
| Underlying DEM data | Copernicus GLO-30 — [open use](https://www.copernicus.eu/en/access-data/copyright-and-licences) | Inherited upstream. |
| Underlying road data | OpenStreetMap (via BBBike), [ODbL 1.0](https://www.openstreetmap.org/copyright) | Inherited upstream; this package qualifies as a *produced work* under ODbL terms. |

## 10. Acknowledgements

Built on the geospatial pipeline documented in [`../preprocess`](../preprocess/README.md). The CARLA precompiled distribution is © 2017–2024 the CARLA Simulator team, licensed under the MIT License; UE4 is © Epic Games. This folder does **not** redistribute CARLA or UE4 binaries.

---

## Contact

Open an issue for technical problems. Research collaboration: `acarrera@air-institute.com`.
