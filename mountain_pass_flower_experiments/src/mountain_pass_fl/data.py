from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .baselines import (
    EV_LIBRARY,
    constant_speed_flat_energy,
    infer_road_condition,
    infer_start_soc_from_text,
    infer_temperature_from_text,
    nominal_energy,
    physics_energy_from_profile,
    standardize_vehicle_key,
)


def discover_data_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    files: list[Path] = []
    for pattern in ("*.csv", "*.CSV"):
        files.extend(data_dir.glob(pattern))
    return sorted(set(files))


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def infer_metadata_from_name(path: Path, scenario_text: str | None = None) -> dict[str, object]:
    text = f"{path.stem} {scenario_text or ''}"
    scenario_match = re.search(r"S\d{2}", text, flags=re.IGNORECASE)
    scenario = scenario_match.group(0).upper() if scenario_match else path.stem
    vehicle_key = standardize_vehicle_key(text)
    road_condition = infer_road_condition(text)
    soc = infer_start_soc_from_text(text)
    temp = infer_temperature_from_text(text)
    return {
        "scenario_code": scenario,
        "vehicle_key_inferred": vehicle_key,
        "road_condition_inferred": road_condition,
        "soc_inferred_percent": soc,
        "ambient_temp_inferred_c": temp,
    }


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Formato no soportado: {path}")


def compute_cumulative_distance(df: pd.DataFrame) -> pd.Series:
    required = ["pos_x", "pos_y", "pos_z"]
    if not all(c in df.columns for c in required):
        raise ValueError("El CSV debe contener pos_x, pos_y y pos_z para calcular distancia acumulada")
    xyz = df[required].astype(float).to_numpy()
    diffs = np.diff(xyz, axis=0)
    d = np.sqrt(np.sum(diffs * diffs, axis=1))
    d = np.insert(d, 0, 0.0)
    return pd.Series(np.cumsum(d), index=df.index, name="cum_dist_m")


def _first_existing(df: pd.DataFrame, cols: list[str], default=None):
    for col in cols:
        if col in df.columns:
            return df[col]
    return default


def _scalar_first(s, default=np.nan):
    if s is None:
        return default
    try:
        if isinstance(s, pd.Series):
            if len(s) == 0:
                return default
            return s.dropna().iloc[0] if len(s.dropna()) else default
        return s
    except Exception:
        return default


def _numeric_mean(group: pd.DataFrame, col: str, default=np.nan) -> float:
    if col not in group.columns:
        return float(default) if default is not None else float("nan")
    s = pd.to_numeric(group[col], errors="coerce")
    if s.notna().any():
        return float(s.mean())
    return float(default) if default is not None else float("nan")


def _numeric_min(group: pd.DataFrame, col: str, default=np.nan) -> float:
    if col not in group.columns:
        return float(default) if default is not None else float("nan")
    s = pd.to_numeric(group[col], errors="coerce")
    if s.notna().any():
        return float(s.min())
    return float(default) if default is not None else float("nan")


def _numeric_max(group: pd.DataFrame, col: str, default=np.nan) -> float:
    if col not in group.columns:
        return float(default) if default is not None else float("nan")
    s = pd.to_numeric(group[col], errors="coerce")
    if s.notna().any():
        return float(s.max())
    return float(default) if default is not None else float("nan")


def _mode_str(group: pd.DataFrame, col: str, default="unknown") -> str:
    if col not in group.columns:
        return default
    s = group[col].dropna().astype(str)
    if len(s) == 0:
        return default
    return str(s.mode().iloc[0]) if not s.mode().empty else str(s.iloc[0])


def prepare_raw_dataframe(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(path)
    df = load_csv(path)
    scenario_text = str(df["scenario_id"].iloc[0]) if "scenario_id" in df.columns and len(df) else path.stem
    meta = infer_metadata_from_name(path, scenario_text=scenario_text)
    meta["source_file"] = path.name
    meta["client_id"] = safe_name(path)

    if "vehicle_key" not in df.columns:
        df["vehicle_key"] = meta["vehicle_key_inferred"]
    else:
        df["vehicle_key"] = df["vehicle_key"].apply(lambda x: standardize_vehicle_key(x, path.stem))

    if "road_condition" not in df.columns:
        df["road_condition"] = meta["road_condition_inferred"]
    else:
        df["road_condition"] = df["road_condition"].fillna(meta["road_condition_inferred"]).apply(infer_road_condition)

    if "ambient_temp_c" not in df.columns:
        df["ambient_temp_c"] = meta["ambient_temp_inferred_c"] if meta["ambient_temp_inferred_c"] is not None else 24.0

    if "soc_percent" not in df.columns:
        df["soc_percent"] = meta["soc_inferred_percent"] if meta["soc_inferred_percent"] is not None else 90.0

    if "dt" not in df.columns:
        if "time_relative" in df.columns:
            dt = pd.to_numeric(df["time_relative"], errors="coerce").diff().fillna(0.05)
            df["dt"] = dt.clip(lower=1e-3, upper=1.0)
        else:
            df["dt"] = 0.05

    for col, default in [
        ("speed_kmh", 0.0),
        ("acc_long", 0.0),
        ("road_grade_deg", 0.0),
        ("road_curvature", 0.0),
        ("wind_long_ms", 0.0),
        ("precip_intensity", 0.0),
        ("battery_temp_c", df["ambient_temp_c"].iloc[0] if len(df) else 24.0),
    ]:
        if col not in df.columns:
            df[col] = default

    if "capacity_usable_wh" not in df.columns:
        vkey = standardize_vehicle_key(df["vehicle_key"].iloc[0], path.stem)
        df["capacity_usable_wh"] = EV_LIBRARY[vkey].usable_wh

    if "energy_used_cum_wh" not in df.columns or "energy_regen_cum_wh" not in df.columns:
        raise ValueError(f"{path.name} no contiene energy_used_cum_wh y energy_regen_cum_wh")

    df = df.copy()
    df["cum_dist_m"] = compute_cumulative_distance(df)
    df["net_energy_cum_wh"] = pd.to_numeric(df["energy_used_cum_wh"], errors="coerce").fillna(0.0) - pd.to_numeric(df["energy_regen_cum_wh"], errors="coerce").fillna(0.0)
    return df, meta


def segment_dataframe(
    df: pd.DataFrame,
    meta: dict[str, object],
    *,
    segment_m: float = 100.0,
    min_segment_m: float = 20.0,
    expected_speed_kmh: float = 35.0,
) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame()

    df = df.sort_values("cum_dist_m").reset_index(drop=True)
    route_distance_m = float(df["cum_dist_m"].iloc[-1] - df["cum_dist_m"].iloc[0])
    if route_distance_m < min_segment_m:
        return pd.DataFrame()

    df["segment_id"] = np.floor(df["cum_dist_m"] / float(segment_m)).astype(int)
    rows: list[dict[str, object]] = []

    client_id = str(meta["client_id"])
    scenario_id = str(_scalar_first(df.get("scenario_id"), meta.get("scenario_code", client_id)))
    vkey = standardize_vehicle_key(_scalar_first(df.get("vehicle_key")), str(meta.get("source_file", "")))

    for seg_id, group in df.groupby("segment_id", sort=True):
        group = group.sort_values("cum_dist_m")
        if len(group) < 2:
            continue
        distance_m = float(group["cum_dist_m"].iloc[-1] - group["cum_dist_m"].iloc[0])
        if distance_m < min_segment_m:
            continue
        duration_s = float(pd.to_numeric(group["dt"], errors="coerce").fillna(0.05).sum())
        net_start = float(group["net_energy_cum_wh"].iloc[0])
        net_end = float(group["net_energy_cum_wh"].iloc[-1])
        energy_true_wh = net_end - net_start

        z = pd.to_numeric(group["pos_z"], errors="coerce").to_numpy(dtype=float)
        dz_steps = np.diff(z) if len(z) > 1 else np.array([0.0])
        delta_z_m = float(z[-1] - z[0]) if len(z) else 0.0
        uphill_m = float(np.sum(np.maximum(dz_steps, 0.0)))
        downhill_m = float(np.sum(np.maximum(-dz_steps, 0.0)))
        horizontal_m = max(1e-6, math.sqrt(max(distance_m**2 - delta_z_m**2, 0.0)))
        grade_from_z_deg = math.degrees(math.atan2(delta_z_m, horizontal_m))

        road_condition = infer_road_condition(_mode_str(group, "road_condition", str(meta.get("road_condition_inferred", "dry"))))
        ambient_temp_c = _numeric_mean(group, "ambient_temp_c", meta.get("ambient_temp_inferred_c", 24.0) or 24.0)

        topo_profile = physics_energy_from_profile(
            speed_kmh=group["speed_kmh"],
            acc_long_ms2=group["acc_long"],
            grade_deg=group["road_grade_deg"],
            dt_s=group["dt"],
            vehicle_key=vkey,
            road_condition=group["road_condition"],
            ambient_temp_c=group["ambient_temp_c"],
            wind_long_ms=group["wind_long_ms"],
            grade_zero=False,
            use_weather=True,
            allow_regen=True,
        )["net_wh"]
        flat_kinematic = physics_energy_from_profile(
            speed_kmh=group["speed_kmh"],
            acc_long_ms2=group["acc_long"],
            grade_deg=0.0,
            dt_s=group["dt"],
            vehicle_key=vkey,
            road_condition=group["road_condition"],
            ambient_temp_c=group["ambient_temp_c"],
            wind_long_ms=group["wind_long_ms"],
            grade_zero=True,
            use_weather=True,
            allow_regen=True,
        )["net_wh"]
        flat_entry = constant_speed_flat_energy(
            distance_m=distance_m,
            vehicle_key=vkey,
            expected_speed_kmh=expected_speed_kmh,
            ambient_temp_c=ambient_temp_c,
            road_condition=road_condition,
            use_weather=False,
        )
        flat_entry_weather = constant_speed_flat_energy(
            distance_m=distance_m,
            vehicle_key=vkey,
            expected_speed_kmh=expected_speed_kmh,
            ambient_temp_c=ambient_temp_c,
            road_condition=road_condition,
            use_weather=True,
        )
        nom = nominal_energy(distance_m, vkey)

        soc_start = float(pd.to_numeric(group["soc_percent"], errors="coerce").iloc[0])
        soc_end = float(pd.to_numeric(group["soc_percent"], errors="coerce").iloc[-1])
        capacity = _numeric_mean(group, "capacity_usable_wh", EV_LIBRARY[vkey].usable_wh)

        rows.append(
            {
                "source_file": meta.get("source_file"),
                "client_id": client_id,
                "scenario_id": scenario_id,
                "scenario_code": meta.get("scenario_code"),
                "vehicle_key": vkey,
                "road_condition": road_condition,
                "segment_id": int(seg_id),
                "segment_start_m": float(group["cum_dist_m"].iloc[0]),
                "segment_end_m": float(group["cum_dist_m"].iloc[-1]),
                "segment_length_m": distance_m,
                "route_distance_m": route_distance_m,
                "distance_from_start_m": float(group["cum_dist_m"].iloc[0]),
                "duration_s": duration_s,
                "altitude_start_m": float(group["pos_z"].iloc[0]),
                "altitude_end_m": float(group["pos_z"].iloc[-1]),
                "delta_z_m": delta_z_m,
                "uphill_m": uphill_m,
                "downhill_m": downhill_m,
                "grade_from_z_deg": grade_from_z_deg,
                "grade_mean_deg": _numeric_mean(group, "road_grade_deg", grade_from_z_deg),
                "grade_max_deg": _numeric_max(group, "road_grade_deg", grade_from_z_deg),
                "grade_min_deg": _numeric_min(group, "road_grade_deg", grade_from_z_deg),
                "curvature_mean": _numeric_mean(group, "road_curvature", 0.0),
                "curvature_abs_mean": float(pd.to_numeric(group["road_curvature"], errors="coerce").abs().mean()) if "road_curvature" in group else 0.0,
                "speed_mean_kmh": _numeric_mean(group, "speed_kmh", expected_speed_kmh),
                "speed_max_kmh": _numeric_max(group, "speed_kmh", expected_speed_kmh),
                "acc_long_mean": _numeric_mean(group, "acc_long", 0.0),
                "acc_long_abs_mean": float(pd.to_numeric(group["acc_long"], errors="coerce").abs().mean()) if "acc_long" in group else 0.0,
                "acc_lat_abs_mean": float(pd.to_numeric(group.get("acc_lat", pd.Series([0.0]*len(group))), errors="coerce").abs().mean()),
                "throttle_mean": _numeric_mean(group, "throttle", 0.0),
                "brake_mean": _numeric_mean(group, "brake", 0.0),
                "steering_abs_mean": float(pd.to_numeric(group.get("steering", pd.Series([0.0]*len(group))), errors="coerce").abs().mean()),
                "soc_start_percent": soc_start,
                "soc_end_percent": soc_end,
                "capacity_usable_wh": capacity,
                "battery_temp_c": _numeric_mean(group, "battery_temp_c", ambient_temp_c),
                "ambient_temp_c": ambient_temp_c,
                "aux_power_w": _numeric_mean(group, "aux_power_w", np.nan),
                "c_rr_eff": _numeric_mean(group, "c_rr_eff", np.nan),
                "road_friction": _numeric_mean(group, "road_friction", np.nan),
                "precip_intensity": _numeric_mean(group, "precip_intensity", 0.0),
                "wind_speed_ms": _numeric_mean(group, "wind_speed_ms", 0.0),
                "wind_long_ms": _numeric_mean(group, "wind_long_ms", 0.0),
                "expected_speed_kmh": expected_speed_kmh,
                "energy_true_wh": energy_true_wh,
                "energy_true_wh_per_km": energy_true_wh / max(distance_m / 1000.0, 1e-6),
                "baseline_nominal_wh": nom,
                "baseline_entry_flat_wh": flat_entry,
                "baseline_entry_flat_weather_wh": flat_entry_weather,
                "baseline_flat_kinematic_wh": flat_kinematic,
                "baseline_topo_physics_wh": topo_profile,
                "residual_entry_flat_wh": energy_true_wh - flat_entry,
                "residual_entry_flat_weather_wh": energy_true_wh - flat_entry_weather,
                "residual_topo_physics_wh": energy_true_wh - topo_profile,
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        # Normalize segment id within each client for sequence-like information without requiring a RNN.
        max_seg = max(1, int(out["segment_id"].max()))
        out["segment_id_norm"] = out["segment_id"] / max_seg
    return out


def build_segments(
    data_dir: str | Path,
    *,
    segment_m: float = 100.0,
    min_segment_m: float = 20.0,
    expected_speed_kmh: float = 35.0,
) -> pd.DataFrame:
    files = discover_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No se han encontrado CSV en {data_dir}")
    tables = []
    errors = []
    for path in files:
        try:
            df, meta = prepare_raw_dataframe(path)
            seg = segment_dataframe(
                df,
                meta,
                segment_m=segment_m,
                min_segment_m=min_segment_m,
                expected_speed_kmh=expected_speed_kmh,
            )
            if len(seg):
                tables.append(seg)
            else:
                errors.append({"file": path.name, "error": "sin segmentos válidos"})
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
    if errors:
        print("Avisos durante la segmentación:")
        for e in errors:
            print(f"  - {e['file']}: {e['error']}")
    if not tables:
        raise RuntimeError("No se generó ningún segmento válido")
    out = pd.concat(tables, ignore_index=True)
    return out


def add_synthetic_clients(
    segments: pd.DataFrame,
    *,
    copies_per_client: int = 0,
    seed: int = 42,
    payload_values_kg: Iterable[float] = (0, 150, 300, 500),
    eta_factors: Iterable[float] = (0.96, 1.0, 1.04),
    crr_factors: Iterable[float] = (0.95, 1.0, 1.10),
    capacity_factors: Iterable[float] = (0.90, 1.0, 1.05),
) -> pd.DataFrame:
    """Create extra simulated clients by perturbing mass/efficiency/capacity.

    This is optional and should be described as synthetic augmentation. It is useful
    when only a few CARLA runs are available and Flower needs a larger number of
    clients to stress-test non-IID aggregation.
    """
    if copies_per_client <= 0:
        return segments.copy()
    rng = np.random.default_rng(seed)
    base = segments.copy()
    synthetic = [base]
    for client_id, g in base.groupby("client_id"):
        for i in range(copies_per_client):
            h = g.copy()
            payload = float(rng.choice(list(payload_values_kg)))
            eta_factor = float(rng.choice(list(eta_factors)))
            crr_factor = float(rng.choice(list(crr_factors)))
            cap_factor = float(rng.choice(list(capacity_factors)))
            new_id = f"{client_id}__syn{i:02d}_payload{int(payload)}"
            h["client_id"] = new_id
            h["source_file"] = str(h["source_file"].iloc[0]) + f"::synthetic::{i}"
            h["payload_kg"] = payload
            h["eta_factor"] = eta_factor
            h["crr_factor"] = crr_factor
            h["capacity_usable_wh"] = h["capacity_usable_wh"] * cap_factor
            # Approximate energy shift due to payload and rolling resistance.
            # Positive uphill consumes more; downhill payload can increase regen.
            dz = h["delta_z_m"].to_numpy(dtype=float)
            dist = h["segment_length_m"].to_numpy(dtype=float)
            eta_drive = 0.90 * eta_factor
            eta_regen = 0.70 * eta_factor
            crr_base = 0.011 * crr_factor
            d_e_grav = np.where(
                dz >= 0,
                payload * 9.80665 * dz / max(eta_drive, 1e-6) / 3600.0,
                payload * 9.80665 * dz * eta_regen / 3600.0,
            )
            d_e_roll = payload * 9.80665 * crr_base * dist / max(eta_drive, 1e-6) / 3600.0
            noise = rng.normal(0.0, 0.02, size=len(h))
            h["energy_true_wh"] = (h["energy_true_wh"] + d_e_grav + d_e_roll) * (1.0 + noise)
            h["energy_true_wh_per_km"] = h["energy_true_wh"] / np.maximum(h["segment_length_m"] / 1000.0, 1e-6)
            # Keep baseline intentionally unaware of payload; residual absorbs it.
            h["residual_entry_flat_wh"] = h["energy_true_wh"] - h["baseline_entry_flat_wh"]
            h["residual_entry_flat_weather_wh"] = h["energy_true_wh"] - h["baseline_entry_flat_weather_wh"]
            h["residual_topo_physics_wh"] = h["energy_true_wh"] - h["baseline_topo_physics_wh"]
            synthetic.append(h)
    return pd.concat(synthetic, ignore_index=True)
