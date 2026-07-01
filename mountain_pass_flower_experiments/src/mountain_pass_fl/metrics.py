from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    bias = float(np.mean(err)) if len(err) else float("nan")
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(err) / denom) * 100.0) if len(err) else float("nan")
    return {"mae_wh": mae, "rmse_wh": rmse, "bias_wh": bias, "mape_percent": mape}


def route_predictions_from_segment_predictions(
    segments: pd.DataFrame,
    segment_pred_wh: Iterable[float],
    *,
    method: str,
) -> pd.DataFrame:
    df = segments.copy()
    df["pred_wh"] = list(segment_pred_wh)
    rows = []
    for client_id, g in df.groupby("client_id"):
        true_wh = float(g["energy_true_wh"].sum())
        pred_wh = float(g["pred_wh"].sum())
        cap = float(g["capacity_usable_wh"].mean())
        rows.append(
            {
                "method": method,
                "client_id": client_id,
                "source_file": str(g["source_file"].iloc[0]),
                "scenario_id": str(g["scenario_id"].iloc[0]),
                "vehicle_key": str(g["vehicle_key"].iloc[0]),
                "road_condition": str(g["road_condition"].iloc[0]),
                "route_distance_m": float(g["segment_length_m"].sum()),
                "capacity_usable_wh": cap,
                "soc_start_percent": float(g["soc_start_percent"].iloc[0]),
                "soc_end_real_percent_from_csv": float(g["soc_end_percent"].iloc[-1]),
                "true_wh": true_wh,
                "pred_wh": pred_wh,
                "error_wh": pred_wh - true_wh,
                "abs_error_wh": abs(pred_wh - true_wh),
                "true_wh_per_km": true_wh / max(float(g["segment_length_m"].sum()) / 1000.0, 1e-6),
                "pred_wh_per_km": pred_wh / max(float(g["segment_length_m"].sum()) / 1000.0, 1e-6),
                "soc_end_pred_percent": float(g["soc_start_percent"].iloc[0]) - pred_wh / max(cap, 1e-6) * 100.0,
                "soc_end_real_percent": float(g["soc_start_percent"].iloc[0]) - true_wh / max(cap, 1e-6) * 100.0,
                "soc_final_error_percent": -(pred_wh - true_wh) / max(cap, 1e-6) * 100.0,
                "n_segments": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def summarize_route_predictions(route_preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in route_preds.groupby("method"):
        s = regression_summary(g["true_wh"].to_numpy(), g["pred_wh"].to_numpy())
        rows.append(
            {
                "method": method,
                **s,
                "mae_kwh": s["mae_wh"] / 1000.0,
                "rmse_kwh": s["rmse_wh"] / 1000.0,
                "mae_soc_final_percent": float(np.mean(np.abs(g["soc_final_error_percent"]))) if len(g) else float("nan"),
                "n_routes": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_wh") if rows else pd.DataFrame()


def baseline_route_predictions(segments: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "B0_nominal_consumption": "baseline_nominal_wh",
        "B1_flat_physics_entry_no_topography": "baseline_entry_flat_wh",
        "B1b_flat_physics_entry_weather_no_topography": "baseline_entry_flat_weather_wh",
        "B2_flat_physics_kinematic_no_topography": "baseline_flat_kinematic_wh",
        "B3_topography_physics_sanity": "baseline_topo_physics_wh",
    }
    tables = []
    for method, col in methods.items():
        if col in segments.columns:
            tables.append(route_predictions_from_segment_predictions(segments, segments[col].to_numpy(), method=method))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def save_metrics(route_preds: pd.DataFrame, out_dir: str | Path, prefix: str) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_path = out_dir / f"{prefix}_route_predictions.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    route_preds.to_csv(route_path, index=False)
    summary = summarize_route_predictions(route_preds)
    summary.to_csv(summary_path, index=False)
    return summary
