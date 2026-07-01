from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .features import FeatureTransformer, TargetScaler
from .metrics import route_predictions_from_segment_predictions, summarize_route_predictions
from .models import ResidualMLP, get_parameters, set_parameters
from .train import predict_residuals, train_torch_model, set_seed


def _average_parameters(old_params: list, new_params: list, alpha: float) -> list:
    return [(1.0 - alpha) * o + alpha * n for o, n in zip(old_params, new_params)]


def run_prequential_beacon(
    segments: pd.DataFrame,
    *,
    client_order: list[str] | None = None,
    feature_set: str,
    target_column: str,
    base_prediction_column: str,
    hidden_layers: Iterable[int] = (64, 64, 32),
    dropout: float = 0.0,
    batch_size: int = 32,
    local_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    aggregation_alpha: float = 1.0,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential predict-then-update experiment.

    This mimics a beacon: vehicle k is evaluated with the model learned from
    previous vehicles, then its local data are used to update the beacon model.
    It uses the same residual MLP/FedAvg logic but runs sequentially to match the
    entry-exit beacon story.
    """
    set_seed(seed)
    if client_order is None:
        client_order = sorted(segments["client_id"].unique().tolist())
    # Normalization statistics are part of the transmitted model package. In a real
    # deployment these can come from design-time calibration. Here they are fit once
    # on all available segments for stable simulation.
    transformer = FeatureTransformer.fit(segments, feature_set=feature_set)
    target_scaler = TargetScaler.fit(segments[target_column].to_numpy(dtype=np.float32))
    global_model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=True)
    route_tables = []
    for order_idx, client_id in enumerate(client_order):
        client_df = segments[segments["client_id"] == client_id].copy()
        if len(client_df) == 0:
            continue
        # 1) Predict before seeing this client's labels.
        residual_pred = predict_residuals(global_model, client_df, transformer, target_scaler)
        segment_pred = client_df[base_prediction_column].to_numpy(dtype=float) + residual_pred
        route_pred = route_predictions_from_segment_predictions(
            client_df,
            segment_pred,
            method="M3_prequential_beacon_residual_mlp",
        )
        route_pred["arrival_order"] = order_idx
        route_pred["n_previous_clients"] = order_idx
        route_tables.append(route_pred)

        # 2) Train locally after exit and aggregate update into beacon model.
        local_model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=False)
        set_parameters(local_model, get_parameters(global_model))
        # Use client data both as train and validation because this is local one-vehicle adaptation.
        train_torch_model(
            local_model,
            client_df,
            client_df,
            transformer,
            target_scaler,
            target_column=target_column,
            batch_size=batch_size,
            epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=0,
        )
        new_params = _average_parameters(get_parameters(global_model), get_parameters(local_model), alpha=aggregation_alpha)
        set_parameters(global_model, new_params)

    route_preds = pd.concat(route_tables, ignore_index=True) if route_tables else pd.DataFrame()
    summary = summarize_route_predictions(route_preds) if len(route_preds) else pd.DataFrame()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        route_preds.to_csv(out_dir / "prequential_route_predictions.csv", index=False)
        summary.to_csv(out_dir / "prequential_summary.csv", index=False)
        torch.save(global_model.state_dict(), out_dir / "prequential_final_model.pt")
        transformer.save(out_dir / "prequential_feature_transformer.json")
        target_scaler.save(out_dir / "prequential_target_scaler.json")
    return route_preds, summary
