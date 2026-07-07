# DEM-to-RoadRunner HD Map middleware

**MATLAB App Designer tool that converts a GeoTIFF Digital Elevation Model (DEM) and an OpenStreetMap road Shapefile into a RoadRunner HD Map (`.rrhd`) with real terrain elevation fused into the road geometry — not draped as a flat mesh underneath it.**

This tool is the geospatial-preprocessing stage of a digital-twin pipeline built jointly by the **AIR Institute** (Salamanca, Spain) and the **Universidad Politécnica de Pachuca / CITNOVA** (Hidalgo, Mexico), for a research project on EV range-risk estimation on low-coverage mountain roads. Its output feeds directly into the CARLA map bundle in [`nous_mountain_carla`](https://github.com/AIRInstitute/nous_mountain_carla), which packages the same case study — the **Puerto de la Quesera** pass (Sistema Central, GU-186 road, Spain) — for simulation.

A scientific manuscript describing the full methodology is in preparation for *Open Research Europe*.

---

## Table of contents

1. [Why this tool exists](#1-why-this-tool-exists)
2. [Inputs](#2-inputs)
3. [Algorithmic pipeline](#3-algorithmic-pipeline)
4. [Requirements](#4-requirements)
5. [Usage](#5-usage)
6. [From `.rrhd` to a CARLA package](#6-from-rrhd-to-a-carla-package)
7. [Known limitations and roadmap](#7-known-limitations-and-roadmap)
8. [Citing this tool](#8-citing-this-tool)
9. [License](#9-license)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Why this tool exists

A widely reported failure mode in the CARLA/RoadRunner community is that a DEM and a road vector layer get combined so that the *visual mesh* carries real terrain, while the *road centerline* itself stays flat (z = 0). That silently invalidates any downstream study that depends on real road grade — energy consumption, braking distance, sensor line-of-sight, etc.

`nous_mountain_preprocess` exists to close that gap: it fuses DEM elevation directly into the 3D geometry of every lane and lane boundary **before** the scene ever reaches RoadRunner, so the exported OpenDRIVE network preserves a non-trivial `<elevationProfile>` for every road.

## 2. Inputs

| Input | Source | Notes |
|---|---|---|
| GeoTIFF DEM tile | [OpenTopography](https://opentopography.org/), Copernicus GLO-30 dataset | ~30 m horizontal resolution, EPSG:4326 |
| Road-network Shapefile | [BBBike extraction service](https://extract.bbbike.org/) (OpenStreetMap data) | Same bounding box as the DEM tile |

For the Puerto de la Quesera case study, the tile used spans DEM bounding box lon `[-3.4001°, -3.3540°]`, lat `[41.1610°, 41.1821°]`, with terrain elevation ranging **1203–1798 m a.s.l.**

## 3. Algorithmic pipeline

1. **Raster cleaning and reference frame.** The GeoTIFF is read with `readgeoraster`; no-data sentinel values (below −10,000) are replaced with `NaN`. The local East-North-Up (ENU) origin used by the generated HD Map is set at the centroid of the DEM bounding box, and stored as the map's `GeoReference`.
2. **Vector-on-raster elevation fusion.** For each polyline in the road Shapefile, the app extracts its (lat, lon) vertices, removes line-separating `NaN`s introduced by `shaperead`, and samples the DEM at each vertex via bicubic interpolation (`geointerp(..., 'Cubic')`). Vertices whose interpolated elevation is `NaN` (outside DEM coverage) are discarded; polylines that retain fewer than two valid vertices are skipped. Surviving vertices are projected to local metric coordinates with a flat-earth approximation around the ENU origin, then smoothed with a length-5 Gaussian window to suppress OpenStreetMap digitization noise.
3. **Lane and boundary construction.** The smoothed centerline of each road becomes the lane axis. Forward and backward lanes are built by offsetting the centerline ±1.75 m (half of a fixed 3.5 m lane width) along the perpendicular unit normal. Each road gets two `roadrunner.hdmap.Lane` objects (forward/backward) and three `roadrunner.hdmap.LaneBoundary` objects (a shared center boundary plus two outer boundaries) — all populated with full 3D geometry (x, y, z), so elevation is native to the map structure, not a post-hoc overlay.
4. **Serialization.** The complete set of lanes and boundaries is assigned to a `roadrunnerHDMap` object and written out as a `.rrhd` file, ready to open directly in RoadRunner with roads already conformed to terrain.

## 4. Requirements

- MATLAB R2024a (App Designer)
- Mapping Toolbox (`readgeoraster`, `geointerp`)
- Automated Driving Toolbox / RoadRunner interoperability classes (`roadrunner.hdmap.*`, `roadrunnerHDMap`)

## 5. Usage

1. Download a DEM tile from OpenTopography (Copernicus GLO-30) covering your road segment of interest.
2. Extract a matching road Shapefile from BBBike for the same bounding box.
3. Launch the app in MATLAB and load both files.
4. Run the pipeline (raster cleaning → elevation fusion → lane construction).
5. Click **Export Road** to export the HD Map (`.rrhd`).
6. Open the `.rrhd` in RoadRunner R2024a for scene finalization.

## 6. From `.rrhd` to a CARLA package

Once you have a `.rrhd`, three more steps (documented in full in [`nous_mountain_carla`](https://github.com/AIRInstitute/nous_mountain_carla)) turn it into a CARLA-importable map:

1. **RoadRunner finalization + CARLA Filmbox export**, producing a `.fbx` mesh, a `.xodr` OpenDRIVE network, and a `.rrdata.xml` metadata file.
2. **UE4 import**, from a *source-built* CARLA workspace: drop the `.fbx + .xodr + .rrdata.xml` set into `Unreal/CarlaUE4/Content/Carla/ExportedMaps/` and run `make import`.
3. **Packaging**: `make package ARGS="--packages=Mountain"` produces a deployable `.tar.gz` that any precompiled CARLA 0.9.16 installation can ingest via `ImportAssets`.

Full instructions, including CARLA server configuration (`config.py -m Mountain --delta-seconds 0.05`), live in the [`nous_mountain_carla/SETUP.md`](https://github.com/AIRInstitute/nous_mountain_carla/blob/main/SETUP.md) guide.

## 7. Known limitations and roadmap

### 7.1. Bounding box entered by hand

The app has no built-in map picker yet. The current workflow requires copying coordinates between OpenTopography and BBBike manually. A future v3 (in Python) will integrate a selector directly.

### 7.2. Lane width hard-coded

`laneWidth = 3.5` (m) is a fixed constant. Mountain roads like the GU-186 are narrower in practice (~5 m total width, ~2.5 m lanes). For now, edit the constant directly in the source; a v3 parameter field is planned.

### 7.3. Single lane pair per road

Every Shapefile road produces exactly one forward and one backward lane — no multi-lane carriageways or shoulders. Fine for rural/mountain segments; insufficient for highways.

### 7.4. Flat-earth projection

The local ENU projection uses a single tangent plane at the DEM centroid. Distortion is negligible (< 1 cm) for tiles under 10 km × 10 km but grows for larger scenes. Use a proper UTM/Lambert projection for continental-scale extraction.

### 7.5. No automated tests

The current implementation is interactive-only. A headless CLI and an end-to-end regression test (DEM → `.rrhd` diffed against a reference output) are on the roadmap.

## 8. Citing this tool

If you use this tool in academic work, please cite:

> Vidal Cuevas, J. E., Mezquita Martín, Y., Valdeolmillos, D., & Carrera González, A. (2026). *nous_mountain_preprocess: a MATLAB App Designer tool for DEM-to-RoadRunner HD Map conversion with elevation-aware road geometry* [Software]. GitHub. https://github.com/AIRInstitute/nous_mountain_preprocess

A `CITATION.cff` file is included for GitHub's automatic citation feature.

## 9. License

MIT — see [`LICENSE`](LICENSE). A permissive license was chosen because this is a code tool, not a data product; it matches common MATLAB-community norms and imposes no obligations on downstream users beyond attribution.

## 10. Acknowledgements

Developed during a research stay at the **AIR Institute**, Salamanca, Spain, in collaboration with the **Universidad Politécnica de Pachuca / CITNOVA**, Mexico. Built on public data from Copernicus (via OpenTopography) and OpenStreetMap contributors (via BBBike).

---

## Contact

Open an issue for technical problems. For research collaboration: `acarrera@air-institute.com`.
