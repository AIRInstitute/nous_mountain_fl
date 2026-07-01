from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features import FeatureTransformer, TargetScaler
from .metrics import route_predictions_from_segment_predictions, summarize_route_predictions
from .models import ResidualMLP, count_parameters


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_clients(
    clients: list[str],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    clients = list(sorted(set(clients)))
    rng = np.random.default_rng(seed)
    rng.shuffle(clients)
    n = len(clients)
    if n == 1:
        return clients, clients, clients
    if n == 2:
        return [clients[0]], [clients[1]], [clients[1]]
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = clients[:n_train]
    val = clients[n_train : n_train + n_val]
    test = clients[n_train + n_val :]
    if not test:
        test = val.copy()
    return train, val, test


def make_dataset(
    df: pd.DataFrame,
    transformer: FeatureTransformer,
    target_scaler: TargetScaler,
    target_column: str,
) -> TensorDataset:
    x = transformer.transform(df)
    y = target_scaler.transform(df[target_column].to_numpy(dtype=np.float32))
    return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))


def train_torch_model(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    transformer: FeatureTransformer,
    target_scaler: TargetScaler,
    *,
    target_column: str,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 25,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, list[float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_ds = make_dataset(train_df, transformer, target_scaler, target_column)
    val_ds = make_dataset(val_df if len(val_df) else train_df, transformer, target_scaler, target_column)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(1, len(train_ds))), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(1, len(val_ds))), shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad = 0
    hist = {"train_loss": [], "val_loss": []}
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).detach().cpu()))
        tr = float(np.mean(losses)) if losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else tr
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va)
        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            print(f"epoch={epoch:04d} train={tr:.5f} val={va:.5f}")
        if va < best_val - 1e-8:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if patience and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def predict_residuals(
    model: torch.nn.Module,
    df: pd.DataFrame,
    transformer: FeatureTransformer,
    target_scaler: TargetScaler,
    *,
    batch_size: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    x = transformer.transform(df)
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            pred_scaled = model(xb).detach().cpu().numpy()
            preds.append(pred_scaled)
    pred_scaled = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    return target_scaler.inverse_transform(pred_scaled)


def run_centralized(
    segments: pd.DataFrame,
    *,
    train_clients: list[str],
    val_clients: list[str],
    test_clients: list[str],
    feature_set: str,
    target_column: str,
    base_prediction_column: str,
    hidden_layers: Iterable[int] = (64, 64, 32),
    dropout: float = 0.0,
    batch_size: int = 32,
    epochs: int = 250,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 35,
    seed: int = 42,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    set_seed(seed)
    train_df = segments[segments["client_id"].isin(train_clients)].copy()
    val_df = segments[segments["client_id"].isin(val_clients)].copy()
    test_df = segments[segments["client_id"].isin(test_clients)].copy()
    if len(train_df) == 0:
        raise RuntimeError("No hay segmentos de entrenamiento")
    if len(val_df) == 0:
        val_df = train_df
    if len(test_df) == 0:
        test_df = val_df

    transformer = FeatureTransformer.fit(train_df, feature_set=feature_set)
    target_scaler = TargetScaler.fit(train_df[target_column].to_numpy(dtype=np.float32))
    model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=True)
    hist = train_torch_model(
        model,
        train_df,
        val_df,
        transformer,
        target_scaler,
        target_column=target_column,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )
    residual_pred = predict_residuals(model, test_df, transformer, target_scaler)
    segment_pred = test_df[base_prediction_column].to_numpy(dtype=float) + residual_pred
    route_preds = route_predictions_from_segment_predictions(test_df, segment_pred, method="M1_centralized_residual_mlp")
    summary = summarize_route_predictions(route_preds)

    artifacts = {
        "model": model,
        "transformer": transformer,
        "target_scaler": target_scaler,
        "history": hist,
        "n_parameters": count_parameters(model),
        "train_clients": train_clients,
        "val_clients": val_clients,
        "test_clients": test_clients,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "centralized_model.pt")
        transformer.save(out_dir / "feature_transformer.json")
        target_scaler.save(out_dir / "target_scaler.json")
        pd.DataFrame(hist).to_csv(out_dir / "centralized_training_history.csv", index=False)
        route_preds.to_csv(out_dir / "centralized_route_predictions.csv", index=False)
        summary.to_csv(out_dir / "centralized_summary.csv", index=False)
    return route_preds, summary, artifacts
