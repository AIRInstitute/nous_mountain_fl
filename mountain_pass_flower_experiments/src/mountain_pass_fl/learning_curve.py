from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .features import FeatureTransformer, TargetScaler
from .metrics import baseline_route_predictions, regression_summary, route_predictions_from_segment_predictions
from .models import ResidualMLP, get_parameters, set_parameters
from .train import predict_residuals, set_seed


def _weighted_average_parameter_sets(param_sets: list[list[np.ndarray]], weights: list[float]) -> list[np.ndarray]:
    if not param_sets:
        raise ValueError("No hay parámetros para agregar")
    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.sum() <= 0:
        weights_arr = np.ones_like(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()
    out: list[np.ndarray] = []
    for layer_idx in range(len(param_sets[0])):
        layer = np.zeros_like(param_sets[0][layer_idx], dtype=np.float64)
        for w, params in zip(weights_arr, param_sets):
            layer += float(w) * params[layer_idx].astype(np.float64)
        out.append(layer.astype(param_sets[0][layer_idx].dtype))
    return out


def _blend_parameters(old_params: list[np.ndarray], new_params: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    alpha = float(alpha)
    return [(1.0 - alpha) * old + alpha * new for old, new in zip(old_params, new_params)]


def _add_lc_columns(
    df: pd.DataFrame,
    *,
    method: str | None = None,
    permutation: int,
    batch_size: int,
    n_seen_clients: int,
    n_remaining_clients: int,
    eval_mode: str,
    backend: str,
    strategy: str,
    order: list[str],
) -> pd.DataFrame:
    out = df.copy()
    if method is not None:
        out["method"] = method
    out["permutation"] = int(permutation)
    out["batch_size"] = int(batch_size)
    out["n_seen_clients"] = int(n_seen_clients)
    out["n_remaining_clients"] = int(n_remaining_clients)
    out["eval_mode"] = eval_mode
    out["backend"] = backend
    out["strategy"] = strategy
    out["arrival_order"] = ";".join(order)
    return out


def _evaluate_residual_model(
    model: ResidualMLP,
    eval_df: pd.DataFrame,
    *,
    transformer: FeatureTransformer,
    target_scaler: TargetScaler,
    base_prediction_column: str,
    method: str,
    permutation: int,
    batch_size: int,
    n_seen_clients: int,
    n_remaining_clients: int,
    eval_mode: str,
    backend: str,
    strategy: str,
    order: list[str],
) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()
    residual_pred = predict_residuals(model, eval_df, transformer, target_scaler)
    segment_pred = eval_df[base_prediction_column].to_numpy(dtype=float) + residual_pred
    routes = route_predictions_from_segment_predictions(eval_df, segment_pred, method=method)
    return _add_lc_columns(
        routes,
        permutation=permutation,
        batch_size=batch_size,
        n_seen_clients=n_seen_clients,
        n_remaining_clients=n_remaining_clients,
        eval_mode=eval_mode,
        backend=backend,
        strategy=strategy,
        order=order,
    )


def _evaluate_baselines(
    eval_df: pd.DataFrame,
    *,
    permutation: int,
    batch_size: int,
    n_seen_clients: int,
    n_remaining_clients: int,
    eval_mode: str,
    order: list[str],
) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()
    routes = baseline_route_predictions(eval_df)
    return _add_lc_columns(
        routes,
        permutation=permutation,
        batch_size=batch_size,
        n_seen_clients=n_seen_clients,
        n_remaining_clients=n_remaining_clients,
        eval_mode=eval_mode,
        backend="baseline",
        strategy="none",
        order=order,
    )


def _train_one_beacon_update(
    global_model: ResidualMLP,
    batch_clients: list[str],
    segments: pd.DataFrame,
    *,
    transformer: FeatureTransformer,
    target_scaler: TargetScaler,
    target_column: str,
    hidden_layers: Iterable[int],
    dropout: float,
    strategy: str,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    proximal_mu: float,
    aggregation_alpha: float,
) -> None:
    # Import here to avoid making the Flower module a hard dependency for users that only run baselines.
    from .flower_exp import _train_local_model

    old_params = get_parameters(global_model)
    local_param_sets: list[list[np.ndarray]] = []
    weights: list[float] = []
    for client_id in batch_clients:
        client_df = segments[segments["client_id"] == client_id].copy()
        if len(client_df) == 0:
            continue
        local_model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=False)
        set_parameters(local_model, old_params)
        _train_local_model(
            local_model,
            client_df,
            transformer,
            target_scaler,
            target_column,
            epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            proximal_mu=proximal_mu if strategy.lower() == "fedprox" else 0.0,
            global_parameters=old_params,
        )
        local_param_sets.append(get_parameters(local_model))
        weights.append(float(len(client_df)))
    if not local_param_sets:
        return
    aggregated = _weighted_average_parameter_sets(local_param_sets, weights)
    if aggregation_alpha < 1.0:
        aggregated = _blend_parameters(old_params, aggregated, alpha=aggregation_alpha)
    set_parameters(global_model, aggregated)


def _parse_seen_counts(values: Iterable[Any] | None, n_clients: int) -> set[int]:
    if not values:
        base = [0, 1, 2, 4, 8, n_clients - 1, n_clients]
    else:
        base = []
        for v in values:
            if isinstance(v, str) and v.lower() == "all":
                base.append(n_clients)
            else:
                try:
                    base.append(int(v))
                except Exception:
                    continue
    return set(sorted({max(0, min(n_clients, int(x))) for x in base}))


def _point_summary(route_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if route_predictions.empty:
        return pd.DataFrame()
    group_cols = ["backend", "strategy", "method", "eval_mode", "batch_size", "permutation", "n_seen_clients"]
    for keys, g in route_predictions.groupby(group_cols, dropna=False):
        backend, strategy, method, eval_mode, batch_size, permutation, n_seen = keys
        s = regression_summary(g["true_wh"].to_numpy(), g["pred_wh"].to_numpy())
        rows.append(
            {
                "backend": backend,
                "strategy": strategy,
                "method": method,
                "eval_mode": eval_mode,
                "batch_size": int(batch_size),
                "permutation": int(permutation),
                "n_seen_clients": int(n_seen),
                **s,
                "mae_kwh": s["mae_wh"] / 1000.0,
                "rmse_kwh": s["rmse_wh"] / 1000.0,
                "mae_soc_final_percent": float(np.mean(np.abs(g["soc_final_error_percent"]))) if len(g) else np.nan,
                "n_routes": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_learning_summary(point_summary: pd.DataFrame) -> pd.DataFrame:
    if point_summary.empty:
        return pd.DataFrame()
    metric_cols = ["mae_wh", "rmse_wh", "bias_wh", "mape_percent", "mae_kwh", "rmse_kwh", "mae_soc_final_percent"]
    rows: list[dict[str, Any]] = []
    group_cols = ["backend", "strategy", "method", "eval_mode", "batch_size", "n_seen_clients"]
    for keys, g in point_summary.groupby(group_cols, dropna=False):
        backend, strategy, method, eval_mode, batch_size, n_seen = keys
        row: dict[str, Any] = {
            "backend": backend,
            "strategy": strategy,
            "method": method,
            "eval_mode": eval_mode,
            "batch_size": int(batch_size),
            "n_seen_clients": int(n_seen),
            "n_permutations": int(g["permutation"].nunique()),
            "mean_eval_routes": float(g["n_routes"].mean()) if len(g) else np.nan,
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
    if not out.empty:
        out = out.sort_values(["eval_mode", "batch_size", "n_seen_clients", "mae_kwh_mean", "method"])
    return out


def run_beacon_learning_curve(
    segments: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None,
    seed: int,
    feature_set: str,
    target_column: str,
    base_prediction_column: str,
    hidden_layers: Iterable[int] = (64, 64, 32),
    dropout: float = 0.0,
    batch_size_train: int = 32,
    local_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    proximal_mu: float = 0.01,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Simulate the beacon learning progressively as vehicles cross the pass.

    This uses sequential/batched FedAvg or FedProx-style aggregation. It is intentionally
    aligned with the V2I story: predict before entering, update after exiting.
    """
    cfg = cfg or {}
    set_seed(seed)
    clients = sorted(segments["client_id"].unique().tolist())
    n_clients = len(clients)
    if n_clients < 2:
        raise RuntimeError("Se necesitan al menos 2 clientes/rutas para curva de aprendizaje")

    permutations = int(cfg.get("permutations", 50))
    batch_sizes = [int(x) for x in cfg.get("batch_sizes", [1, 2, 4])]
    strategies = [str(x).lower() for x in cfg.get("strategies", ["fedavg", "fedprox"])]
    include_baselines = bool(cfg.get("include_baselines", True))
    eval_on_remaining = bool(cfg.get("eval_on_remaining", True))
    eval_on_all_seen = bool(cfg.get("eval_on_all_seen", True))
    aggregation_alpha = float(cfg.get("aggregation_alpha", 1.0))
    checkpoint_counts = _parse_seen_counts(cfg.get("seen_counts", None), n_clients)

    # The normalization package is part of the model transmitted by the beacon. We fit it once
    # on the available simulated corpus for stable experiments; target mean remains zero.
    transformer = FeatureTransformer.fit(segments, feature_set=feature_set)
    target_scaler = TargetScaler.fit(segments[target_column].to_numpy(dtype=np.float32))

    route_tables: list[pd.DataFrame] = []
    skipped: list[str] = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for perm_idx in range(permutations):
        rng = np.random.default_rng(seed + 50_000 + perm_idx)
        order = clients.copy()
        rng.shuffle(order)
        # Make the first permutation deterministic/readable for debugging.
        if perm_idx == 0 and bool(cfg.get("first_permutation_sorted", True)):
            order = clients.copy()

        for bsz in batch_sizes:
            for strategy in strategies:
                global_model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=True)
                seen = 0
                while True:
                    should_eval = seen in checkpoint_counts or seen == 0 or seen == n_clients
                    if should_eval:
                        remaining_clients = order[seen:]
                        seen_clients = order[:seen]
                        if eval_on_remaining and remaining_clients:
                            eval_df = segments[segments["client_id"].isin(remaining_clients)].copy()
                            if include_baselines:
                                route_tables.append(
                                    _evaluate_baselines(
                                        eval_df,
                                        permutation=perm_idx,
                                        batch_size=bsz,
                                        n_seen_clients=seen,
                                        n_remaining_clients=len(remaining_clients),
                                        eval_mode="remaining_unseen",
                                        order=order,
                                    )
                                )
                            route_tables.append(
                                _evaluate_residual_model(
                                    global_model,
                                    eval_df,
                                    transformer=transformer,
                                    target_scaler=target_scaler,
                                    base_prediction_column=base_prediction_column,
                                    method=f"LC_seq_{strategy}_beacon_residual_mlp",
                                    permutation=perm_idx,
                                    batch_size=bsz,
                                    n_seen_clients=seen,
                                    n_remaining_clients=len(remaining_clients),
                                    eval_mode="remaining_unseen",
                                    backend="sequential_beacon",
                                    strategy=strategy,
                                    order=order,
                                )
                            )
                        if eval_on_all_seen and (seen == n_clients or seen in checkpoint_counts):
                            # Replay-style diagnostic: evaluates all routes with the model after seeing `seen` clients.
                            # This is not a pure generalization metric when seen > 0, so eval_mode labels it explicitly.
                            eval_df_all = segments.copy()
                            if include_baselines:
                                route_tables.append(
                                    _evaluate_baselines(
                                        eval_df_all,
                                        permutation=perm_idx,
                                        batch_size=bsz,
                                        n_seen_clients=seen,
                                        n_remaining_clients=max(0, n_clients - seen),
                                        eval_mode="all_routes_replay",
                                        order=order,
                                    )
                                )
                            route_tables.append(
                                _evaluate_residual_model(
                                    global_model,
                                    eval_df_all,
                                    transformer=transformer,
                                    target_scaler=target_scaler,
                                    base_prediction_column=base_prediction_column,
                                    method=f"LC_seq_{strategy}_beacon_residual_mlp",
                                    permutation=perm_idx,
                                    batch_size=bsz,
                                    n_seen_clients=seen,
                                    n_remaining_clients=max(0, n_clients - seen),
                                    eval_mode="all_routes_replay",
                                    backend="sequential_beacon",
                                    strategy=strategy,
                                    order=order,
                                )
                            )
                    if seen >= n_clients:
                        break
                    batch_clients = order[seen : min(n_clients, seen + bsz)]
                    try:
                        _train_one_beacon_update(
                            global_model,
                            batch_clients,
                            segments,
                            transformer=transformer,
                            target_scaler=target_scaler,
                            target_column=target_column,
                            hidden_layers=hidden_layers,
                            dropout=dropout,
                            strategy=strategy,
                            local_epochs=local_epochs,
                            batch_size=batch_size_train,
                            lr=lr,
                            weight_decay=weight_decay,
                            proximal_mu=proximal_mu,
                            aggregation_alpha=aggregation_alpha,
                        )
                    except Exception as exc:
                        skipped.append(f"perm={perm_idx} batch={bsz} strategy={strategy} seen={seen}: {exc}")
                        break
                    seen += len(batch_clients)

            if out_dir is not None and route_tables:
                pd.concat(route_tables, ignore_index=True).to_csv(Path(out_dir) / "learning_curve_route_predictions_partial.csv", index=False)

    route_predictions = pd.concat(route_tables, ignore_index=True) if route_tables else pd.DataFrame()
    point_summary = _point_summary(route_predictions)
    summary = _aggregate_learning_summary(point_summary)
    if out_dir is not None:
        out_dir = Path(out_dir)
        route_predictions.to_csv(out_dir / "learning_curve_route_predictions.csv", index=False)
        point_summary.to_csv(out_dir / "learning_curve_point_summary.csv", index=False)
        summary.to_csv(out_dir / "learning_curve_summary.csv", index=False)
        pd.DataFrame({"skipped": skipped}).to_csv(out_dir / "learning_curve_skipped.csv", index=False)
    return route_predictions, point_summary, summary, skipped


def run_flower_learning_checkpoints(
    segments: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None,
    seed: int,
    feature_set: str,
    target_column: str,
    base_prediction_column: str,
    hidden_layers: Iterable[int] = (64, 64, 32),
    dropout: float = 0.0,
    batch_size_train: int = 32,
    local_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    proximal_mu: float = 0.01,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Optional real-Flower checkpoints: train with the first k clients and evaluate remaining ones.

    This is heavier than the sequential curve because it starts a Flower simulation for each
    checkpoint, but it is useful to explicitly show Flower behavior after 2, 4, ... vehicles.
    """
    cfg = cfg or {}
    if not bool(cfg.get("enabled", False)):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    from .flower_exp import run_flower

    clients = sorted(segments["client_id"].unique().tolist())
    n_clients = len(clients)
    permutations = int(cfg.get("permutations", 1))
    strategies = [str(x).lower() for x in cfg.get("strategies", ["fedavg"])]
    seen_counts = sorted(_parse_seen_counts(cfg.get("seen_counts", [2, 4, 8, "all"]), n_clients))
    seen_counts = [x for x in seen_counts if x >= 2]
    rounds = int(cfg.get("rounds", 8))
    include_baselines = bool(cfg.get("include_baselines", True))
    route_tables: list[pd.DataFrame] = []
    skipped: list[str] = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for perm_idx in range(permutations):
        rng = np.random.default_rng(seed + 80_000 + perm_idx)
        order = clients.copy()
        rng.shuffle(order)
        if perm_idx == 0 and bool(cfg.get("first_permutation_sorted", True)):
            order = clients.copy()
        for n_seen in seen_counts:
            n_seen = min(int(n_seen), n_clients)
            train_clients = order[:n_seen]
            remaining_clients = order[n_seen:]
            if remaining_clients:
                eval_clients = remaining_clients
                eval_mode = "remaining_unseen"
            else:
                eval_clients = clients
                eval_mode = "all_routes_replay"
            val_clients = train_clients[-max(1, min(2, len(train_clients))) :]
            eval_df = segments[segments["client_id"].isin(eval_clients)].copy()
            if include_baselines:
                route_tables.append(
                    _evaluate_baselines(
                        eval_df,
                        permutation=perm_idx,
                        batch_size=max(1, n_seen),
                        n_seen_clients=n_seen,
                        n_remaining_clients=len(remaining_clients),
                        eval_mode=eval_mode,
                        order=order,
                    )
                )
            for strategy in strategies:
                try:
                    print(f"  Flower checkpoint perm={perm_idx} seen={n_seen}/{n_clients} strategy={strategy}")
                    routes, _, _ = run_flower(
                        segments,
                        train_clients=train_clients,
                        val_clients=val_clients,
                        test_clients=eval_clients,
                        strategy_name=strategy,
                        feature_set=feature_set,
                        target_column=target_column,
                        base_prediction_column=base_prediction_column,
                        hidden_layers=hidden_layers,
                        dropout=dropout,
                        batch_size=batch_size_train,
                        rounds=rounds,
                        local_epochs=local_epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        seed=seed + perm_idx + n_seen,
                        fraction_fit=float(cfg.get("fraction_fit", 1.0)),
                        fraction_evaluate=float(cfg.get("fraction_evaluate", 1.0)),
                        min_fit_clients=min(int(cfg.get("min_fit_clients", 2)), len(train_clients)),
                        min_evaluate_clients=min(int(cfg.get("min_evaluate_clients", 1)), len(train_clients)),
                        min_available_clients=min(int(cfg.get("min_available_clients", 2)), len(train_clients)),
                        proximal_mu=proximal_mu,
                        out_dir=None,
                    )
                    routes = _add_lc_columns(
                        routes,
                        method=f"LC_flower_{strategy}_checkpoint_residual_mlp",
                        permutation=perm_idx,
                        batch_size=max(1, n_seen),
                        n_seen_clients=n_seen,
                        n_remaining_clients=len(remaining_clients),
                        eval_mode=eval_mode,
                        backend="flower_checkpoint",
                        strategy=strategy,
                        order=order,
                    )
                    route_tables.append(routes)
                except Exception as exc:
                    skipped.append(f"flower_checkpoint perm={perm_idx} seen={n_seen} strategy={strategy}: {exc}")
            if out_dir is not None and route_tables:
                pd.concat(route_tables, ignore_index=True).to_csv(Path(out_dir) / "flower_checkpoints_route_predictions_partial.csv", index=False)

    route_predictions = pd.concat(route_tables, ignore_index=True) if route_tables else pd.DataFrame()
    point_summary = _point_summary(route_predictions)
    summary = _aggregate_learning_summary(point_summary)
    if out_dir is not None:
        out_dir = Path(out_dir)
        route_predictions.to_csv(out_dir / "flower_checkpoints_route_predictions.csv", index=False)
        point_summary.to_csv(out_dir / "flower_checkpoints_point_summary.csv", index=False)
        summary.to_csv(out_dir / "flower_checkpoints_summary.csv", index=False)
        pd.DataFrame({"skipped": skipped}).to_csv(out_dir / "flower_checkpoints_skipped.csv", index=False)
    return route_predictions, point_summary, summary, skipped
