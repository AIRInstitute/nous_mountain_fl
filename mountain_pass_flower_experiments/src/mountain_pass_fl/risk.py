from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METADATA_COLUMNS = [
    "protocol",
    "fold_id",
    "fold_index",
    "split_seed",
    "backend",
    "strategy",
    "eval_mode",
    "batch_size",
    "permutation",
    "n_seen_clients",
    "n_remaining_clients",
]


def make_risk_table(
    route_predictions: pd.DataFrame,
    *,
    soc_grid_percent: list[float],
    reserve_soc_percent: float = 5.0,
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    metadata_columns = metadata_columns if metadata_columns is not None else DEFAULT_METADATA_COLUMNS
    rows = []
    for _, r in route_predictions.iterrows():
        cap = float(r["capacity_usable_wh"])
        true_wh = float(r["true_wh"])
        pred_wh = float(r["pred_wh"])
        meta = {col: r[col] for col in metadata_columns if col in route_predictions.columns}
        for soc in soc_grid_percent:
            available_wh = cap * float(soc) / 100.0
            reserve_wh = cap * float(reserve_soc_percent) / 100.0
            real_safe = (available_wh - true_wh) >= reserve_wh
            pred_safe = (available_wh - pred_wh) >= reserve_wh
            rows.append(
                {
                    **meta,
                    "method": r["method"],
                    "client_id": r["client_id"],
                    "vehicle_key": r.get("vehicle_key", "unknown"),
                    "road_condition": r.get("road_condition", "unknown"),
                    "soc_start_percent_test": float(soc),
                    "reserve_soc_percent": float(reserve_soc_percent),
                    "true_wh": true_wh,
                    "pred_wh": pred_wh,
                    "real_safe": bool(real_safe),
                    "pred_safe": bool(pred_safe),
                    "false_safe": bool(pred_safe and not real_safe),
                    "false_warning": bool((not pred_safe) and real_safe),
                    "soc_final_real_percent": float(soc) - true_wh / max(cap, 1e-6) * 100.0,
                    "soc_final_pred_percent": float(soc) - pred_wh / max(cap, 1e-6) * 100.0,
                }
            )
    return pd.DataFrame(rows)


def _risk_counts(g: pd.DataFrame) -> dict[str, float | int]:
    tp = int((~g["pred_safe"] & ~g["real_safe"]).sum())  # correctly warns risk
    fp = int((~g["pred_safe"] & g["real_safe"]).sum())   # false warning
    fn = int((g["pred_safe"] & ~g["real_safe"]).sum())   # false safe
    tn = int((g["pred_safe"] & g["real_safe"]).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "n_decisions": int(len(g)),
        "false_safe_rate": float(g["false_safe"].mean()) if len(g) else np.nan,
        "false_warning_rate": float(g["false_warning"].mean()) if len(g) else np.nan,
        "risk_precision": precision,
        "risk_recall": recall,
        "accuracy": float((g["pred_safe"] == g["real_safe"]).mean()) if len(g) else np.nan,
        "tp_warn_risk": tp,
        "fp_false_warning": fp,
        "fn_false_safe": fn,
        "tn_safe": tn,
    }


def summarize_risk(risk_table: pd.DataFrame) -> pd.DataFrame:
    return summarize_risk_by(risk_table, group_cols=["method"])


def summarize_risk_by(risk_table: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if risk_table.empty:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in risk_table.columns]
    if not group_cols:
        group_cols = ["method"]
    for keys, g in risk_table.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(_risk_counts(g))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "false_safe_rate" in out.columns:
        out = out.sort_values(group_cols[:-1] + ["false_safe_rate"] if len(group_cols) > 1 else ["false_safe_rate"])
    return out


def run_risk_analysis(route_predictions: pd.DataFrame, out_dir: str | Path, *, soc_grid_percent: list[float], reserve_soc_percent: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_table = make_risk_table(route_predictions, soc_grid_percent=soc_grid_percent, reserve_soc_percent=reserve_soc_percent)
    risk_summary = summarize_risk(risk_table)
    risk_table.to_csv(out_dir / "risk_decisions.csv", index=False)
    risk_summary.to_csv(out_dir / "risk_summary.csv", index=False)
    return risk_table, risk_summary
