from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .metrics import baseline_route_predictions, summarize_route_predictions, regression_summary
from .train import run_centralized, split_clients


def client_metadata(segments: pd.DataFrame) -> pd.DataFrame:
    """Return one row per client/route with grouping labels used by CV."""
    rows: list[dict[str, Any]] = []
    for client_id, g in segments.groupby("client_id"):
        vehicle = str(g["vehicle_key"].iloc[0]) if "vehicle_key" in g.columns else "unknown"
        road = str(g["road_condition"].iloc[0]) if "road_condition" in g.columns else "unknown"
        scenario = str(g["scenario_id"].iloc[0]) if "scenario_id" in g.columns else str(client_id)
        scenario_code = str(g["scenario_code"].iloc[0]) if "scenario_code" in g.columns else scenario
        rows.append(
            {
                "client_id": str(client_id),
                "source_file": str(g["source_file"].iloc[0]) if "source_file" in g.columns else str(client_id),
                "scenario_id": scenario,
                "scenario_code": scenario_code,
                "vehicle_key": vehicle,
                "road_condition": road,
                "scenario_group": f"{vehicle}__{road}",
                "route_distance_m": float(g["segment_length_m"].sum()),
                "n_segments": int(len(g)),
                "true_wh": float(g["energy_true_wh"].sum()) if "energy_true_wh" in g.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)


def filter_segments_by_min_route_distance(segments: pd.DataFrame, min_route_distance_m: float | None) -> pd.DataFrame:
    """Optionally remove short/partial routes from the experimental set."""
    if not min_route_distance_m or min_route_distance_m <= 0:
        return segments.copy()
    meta = client_metadata(segments)
    keep = meta.loc[meta["route_distance_m"] >= float(min_route_distance_m), "client_id"].tolist()
    return segments[segments["client_id"].isin(keep)].copy()


def _make_train_val_from_remaining(
    all_clients: Iterable[str],
    test_clients: Iterable[str],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    all_clients = list(sorted(set(str(c) for c in all_clients)))
    test_clients = list(sorted(set(str(c) for c in test_clients)))
    remaining = [c for c in all_clients if c not in set(test_clients)]
    if not remaining:
        return [], [], test_clients
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    if len(remaining) == 1:
        return remaining, remaining, test_clients
    n_val = max(1, int(round(len(remaining) * float(val_fraction))))
    if n_val >= len(remaining):
        n_val = max(1, len(remaining) - 1)
    val = sorted(remaining[:n_val])
    train = sorted(remaining[n_val:])
    if not train:
        train = sorted(remaining)
    return train, val, test_clients


def _add_split_columns(
    df: pd.DataFrame,
    *,
    protocol: str,
    fold_id: str,
    fold_index: int,
    train_clients: list[str],
    val_clients: list[str],
    test_clients: list[str],
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    out["protocol"] = protocol
    out["fold_id"] = fold_id
    out["fold_index"] = int(fold_index)
    out["split_seed"] = int(seed)
    out["n_train_clients"] = int(len(train_clients))
    out["n_val_clients"] = int(len(val_clients))
    out["n_test_clients"] = int(len(test_clients))
    out["train_clients"] = ";".join(train_clients)
    out["val_clients"] = ";".join(val_clients)
    out["test_clients"] = ";".join(test_clients)
    return out


def generate_cv_folds(
    segments: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build route-level CV folds.

    The split unit is always the full CSV/client. No segment-level random split is used.
    """
    cfg = cfg or {}
    meta = client_metadata(segments)
    clients = meta["client_id"].tolist()
    val_fraction = float(cfg.get("val_fraction", 0.15))
    folds: list[dict[str, Any]] = []

    protocols = cfg.get("protocols", {}) or {}
    fold_counter = 0

    if protocols.get("leave_one_route", True):
        for client in clients:
            tr, va, te = _make_train_val_from_remaining(clients, [client], val_fraction=val_fraction, seed=seed + fold_counter)
            folds.append(
                {
                    "protocol": "leave_one_route_out",
                    "fold_id": f"loro_{fold_counter:03d}_{client}",
                    "fold_index": fold_counter,
                    "train_clients": tr,
                    "val_clients": va,
                    "test_clients": te,
                    "seed": seed + fold_counter,
                }
            )
            fold_counter += 1

    if protocols.get("leave_one_scenario_group", True):
        for group_name, g in meta.groupby("scenario_group"):
            test = g["client_id"].tolist()
            if len(test) == len(clients):
                continue
            tr, va, te = _make_train_val_from_remaining(clients, test, val_fraction=val_fraction, seed=seed + fold_counter)
            folds.append(
                {
                    "protocol": "leave_one_scenario_group_out",
                    "fold_id": f"losg_{fold_counter:03d}_{group_name}",
                    "fold_index": fold_counter,
                    "train_clients": tr,
                    "val_clients": va,
                    "test_clients": te,
                    "group_value": str(group_name),
                    "seed": seed + fold_counter,
                }
            )
            fold_counter += 1

    if protocols.get("leave_one_weather", True):
        for weather, g in meta.groupby("road_condition"):
            test = g["client_id"].tolist()
            if len(test) == len(clients):
                continue
            tr, va, te = _make_train_val_from_remaining(clients, test, val_fraction=val_fraction, seed=seed + fold_counter)
            folds.append(
                {
                    "protocol": "leave_one_weather_out",
                    "fold_id": f"loweather_{fold_counter:03d}_{weather}",
                    "fold_index": fold_counter,
                    "train_clients": tr,
                    "val_clients": va,
                    "test_clients": te,
                    "group_value": str(weather),
                    "seed": seed + fold_counter,
                }
            )
            fold_counter += 1

    if protocols.get("leave_one_vehicle", True):
        for vehicle, g in meta.groupby("vehicle_key"):
            test = g["client_id"].tolist()
            if len(test) == len(clients):
                continue
            tr, va, te = _make_train_val_from_remaining(clients, test, val_fraction=val_fraction, seed=seed + fold_counter)
            folds.append(
                {
                    "protocol": "leave_one_vehicle_out",
                    "fold_id": f"lovehicle_{fold_counter:03d}_{vehicle}",
                    "fold_index": fold_counter,
                    "train_clients": tr,
                    "val_clients": va,
                    "test_clients": te,
                    "group_value": str(vehicle),
                    "seed": seed + fold_counter,
                }
            )
            fold_counter += 1

    rr_cfg = protocols.get("repeated_random", {}) or {}
    if rr_cfg.get("enabled", True):
        n_splits = int(rr_cfg.get("n_splits", 20))
        train_fraction = float(rr_cfg.get("train_fraction", 0.70))
        val_fraction_rr = float(rr_cfg.get("val_fraction", 0.15))
        test_fraction = float(rr_cfg.get("test_fraction", 0.15))
        for i in range(n_splits):
            tr, va, te = split_clients(
                clients,
                train_fraction=train_fraction,
                val_fraction=val_fraction_rr,
                test_fraction=test_fraction,
                seed=seed + 10_000 + i,
            )
            folds.append(
                {
                    "protocol": "repeated_random_client_split",
                    "fold_id": f"rr_{i:03d}",
                    "fold_index": fold_counter,
                    "train_clients": tr,
                    "val_clients": va,
                    "test_clients": te,
                    "seed": seed + 10_000 + i,
                }
            )
            fold_counter += 1

    max_folds = cfg.get("max_folds", None)
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    return folds


def _fold_summaries(route_preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if route_preds.empty:
        return pd.DataFrame()
    group_cols = ["protocol", "fold_id", "fold_index", "method"]
    for keys, g in route_preds.groupby(group_cols, dropna=False):
        protocol, fold_id, fold_index, method = keys
        s = regression_summary(g["true_wh"].to_numpy(), g["pred_wh"].to_numpy())
        rows.append(
            {
                "protocol": protocol,
                "fold_id": fold_id,
                "fold_index": int(fold_index),
                "method": method,
                **s,
                "mae_kwh": s["mae_wh"] / 1000.0,
                "rmse_kwh": s["rmse_wh"] / 1000.0,
                "mae_soc_final_percent": float(np.mean(np.abs(g["soc_final_error_percent"]))) if len(g) else np.nan,
                "n_routes": int(len(g)),
                "n_train_clients": int(g["n_train_clients"].iloc[0]) if "n_train_clients" in g else np.nan,
                "n_test_clients": int(g["n_test_clients"].iloc[0]) if "n_test_clients" in g else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_cv_summary(fold_summary: pd.DataFrame) -> pd.DataFrame:
    if fold_summary.empty:
        return pd.DataFrame()
    metric_cols = [
        "mae_wh",
        "rmse_wh",
        "bias_wh",
        "mape_percent",
        "mae_kwh",
        "rmse_kwh",
        "mae_soc_final_percent",
    ]
    rows: list[dict[str, Any]] = []
    for (protocol, method), g in fold_summary.groupby(["protocol", "method"], dropna=False):
        row: dict[str, Any] = {
            "protocol": protocol,
            "method": method,
            "n_folds": int(len(g)),
            "total_test_routes": int(g["n_routes"].sum()),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals):
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=0))
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_q25"] = float(vals.quantile(0.25))
                row[f"{col}_q75"] = float(vals.quantile(0.75))
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_q25"] = np.nan
                row[f"{col}_q75"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "mae_kwh_mean" in out.columns:
        out = out.sort_values(["protocol", "mae_kwh_mean", "method"])
    return out


def run_cross_validation(
    segments: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None,
    seed: int,
    feature_set: str,
    target_column: str,
    base_prediction_column: str,
    hidden_layers: Iterable[int] = (64, 64, 32),
    dropout: float = 0.0,
    batch_size: int = 32,
    centralized_epochs: int = 160,
    local_epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 25,
    run_centralized_models: bool = True,
    run_flower_models: bool = False,
    flower_strategies: Iterable[str] = ("fedavg",),
    flower_rounds: int = 8,
    flower_fraction_fit: float = 1.0,
    flower_fraction_evaluate: float = 1.0,
    flower_min_fit_clients: int = 2,
    flower_min_evaluate_clients: int = 1,
    flower_min_available_clients: int = 2,
    flower_proximal_mu: float = 0.01,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Run route-level CV experiments.

    Returns route-level predictions, fold-level metrics, aggregate metrics and skipped/error messages.
    """
    cfg = cfg or {}
    folds = generate_cv_folds(segments, cfg=cfg, seed=seed)
    skipped: list[str] = []
    route_tables: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    for idx, fold in enumerate(folds):
        protocol = str(fold["protocol"])
        fold_id = str(fold["fold_id"])
        tr = list(fold["train_clients"])
        va = list(fold["val_clients"])
        te = list(fold["test_clients"])
        fold_seed = int(fold.get("seed", seed + idx))
        split_rows.append(
            {
                "protocol": protocol,
                "fold_id": fold_id,
                "fold_index": int(fold["fold_index"]),
                "seed": fold_seed,
                "group_value": fold.get("group_value", ""),
                "n_train_clients": len(tr),
                "n_val_clients": len(va),
                "n_test_clients": len(te),
                "train_clients": ";".join(tr),
                "val_clients": ";".join(va),
                "test_clients": ";".join(te),
            }
        )
        if len(tr) == 0 or len(te) == 0:
            skipped.append(f"{protocol}/{fold_id}: split sin train o test")
            continue
        print(f"  CV {idx + 1}/{len(folds)}: {protocol} | test={len(te)} train={len(tr)}")
        test_df = segments[segments["client_id"].isin(te)].copy()
        try:
            base_routes = baseline_route_predictions(test_df)
            base_routes = _add_split_columns(
                base_routes,
                protocol=protocol,
                fold_id=fold_id,
                fold_index=int(fold["fold_index"]),
                train_clients=tr,
                val_clients=va,
                test_clients=te,
                seed=fold_seed,
            )
            route_tables.append(base_routes)
        except Exception as exc:
            skipped.append(f"Baselines {protocol}/{fold_id}: {exc}")

        if run_centralized_models:
            try:
                routes, _, _ = run_centralized(
                    segments,
                    train_clients=tr,
                    val_clients=va,
                    test_clients=te,
                    feature_set=feature_set,
                    target_column=target_column,
                    base_prediction_column=base_prediction_column,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    batch_size=batch_size,
                    epochs=centralized_epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    patience=patience,
                    seed=fold_seed,
                    out_dir=None,
                )
                routes = _add_split_columns(
                    routes,
                    protocol=protocol,
                    fold_id=fold_id,
                    fold_index=int(fold["fold_index"]),
                    train_clients=tr,
                    val_clients=va,
                    test_clients=te,
                    seed=fold_seed,
                )
                route_tables.append(routes)
            except Exception as exc:
                skipped.append(f"Centralized {protocol}/{fold_id}: {exc}")

        if run_flower_models:
            try:
                from .flower_exp import run_flower

                for strategy in flower_strategies:
                    try:
                        routes, _, _ = run_flower(
                            segments,
                            train_clients=tr,
                            val_clients=va,
                            test_clients=te,
                            strategy_name=str(strategy),
                            feature_set=feature_set,
                            target_column=target_column,
                            base_prediction_column=base_prediction_column,
                            hidden_layers=hidden_layers,
                            dropout=dropout,
                            batch_size=batch_size,
                            rounds=flower_rounds,
                            local_epochs=local_epochs,
                            lr=lr,
                            weight_decay=weight_decay,
                            seed=fold_seed,
                            fraction_fit=flower_fraction_fit,
                            fraction_evaluate=flower_fraction_evaluate,
                            min_fit_clients=flower_min_fit_clients,
                            min_evaluate_clients=flower_min_evaluate_clients,
                            min_available_clients=flower_min_available_clients,
                            proximal_mu=flower_proximal_mu,
                            out_dir=None,
                        )
                        routes = _add_split_columns(
                            routes,
                            protocol=protocol,
                            fold_id=fold_id,
                            fold_index=int(fold["fold_index"]),
                            train_clients=tr,
                            val_clients=va,
                            test_clients=te,
                            seed=fold_seed,
                        )
                        route_tables.append(routes)
                    except Exception as exc:
                        skipped.append(f"Flower {strategy} {protocol}/{fold_id}: {exc}")
            except Exception as exc:
                skipped.append(f"Flower import {protocol}/{fold_id}: {exc}")

        if out_dir is not None and route_tables:
            # Save progress after every fold so a long Windows/Ray run can be inspected if interrupted.
            pd.concat(route_tables, ignore_index=True).to_csv(Path(out_dir) / "cv_route_predictions_partial.csv", index=False)

    route_predictions = pd.concat(route_tables, ignore_index=True) if route_tables else pd.DataFrame()
    fold_summary = _fold_summaries(route_predictions)
    cv_summary = _aggregate_cv_summary(fold_summary)
    if out_dir is not None:
        out_dir = Path(out_dir)
        route_predictions.to_csv(out_dir / "cv_route_predictions.csv", index=False)
        fold_summary.to_csv(out_dir / "cv_fold_summary.csv", index=False)
        cv_summary.to_csv(out_dir / "cv_summary.csv", index=False)
        pd.DataFrame(split_rows).to_csv(out_dir / "cv_splits.csv", index=False)
        pd.DataFrame({"skipped": skipped}).to_csv(out_dir / "cv_skipped.csv", index=False)
    return route_predictions, fold_summary, cv_summary, skipped
