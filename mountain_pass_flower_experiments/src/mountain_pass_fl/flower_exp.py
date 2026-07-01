from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .features import FeatureTransformer, TargetScaler
from .metrics import route_predictions_from_segment_predictions, summarize_route_predictions
from .models import ResidualMLP, get_parameters, set_parameters, count_parameters
from .train import make_dataset, predict_residuals, set_seed


class EVFlowerClient:  # subclassed dynamically to avoid importing flwr when unavailable
    pass


def _train_local_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    transformer: FeatureTransformer,
    target_scaler: TargetScaler,
    target_column: str,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    proximal_mu: float | None = None,
    global_parameters: list | None = None,
) -> float:
    ds = make_dataset(df, transformer, target_scaler, target_column)
    loader = DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    global_tensors = None
    if proximal_mu is not None and proximal_mu > 0 and global_parameters is not None:
        global_tensors = [torch.tensor(p, dtype=param.dtype, device=device) for p, param in zip(global_parameters, model.parameters())]
    last_loss = 0.0
    for _ in range(int(epochs)):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if global_tensors is not None:
                prox = torch.tensor(0.0, device=device)
                for p, g in zip(model.parameters(), global_tensors):
                    prox = prox + torch.sum((p - g) ** 2)
                loss = loss + (float(proximal_mu) / 2.0) * prox
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if losses:
            last_loss = float(np.mean(losses))
    return last_loss


def run_flower(
    segments: pd.DataFrame,
    *,
    train_clients: list[str],
    val_clients: list[str],
    test_clients: list[str],
    strategy_name: str,
    feature_set: str,
    target_column: str,
    base_prediction_column: str,
    hidden_layers: Iterable[int] = (64, 64, 32),
    dropout: float = 0.0,
    batch_size: int = 32,
    rounds: int = 20,
    local_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 1,
    min_available_clients: int = 2,
    proximal_mu: float = 0.01,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run a Flower simulation and evaluate the final global model on test clients."""
    try:
        import flwr as fl
        from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Flower no está instalado. Ejecuta: pip install -r requirements.txt "
            "o usa --skip-flower para omitir este experimento."
        ) from exc

    set_seed(seed)
    train_df = segments[segments["client_id"].isin(train_clients)].copy()
    val_df = segments[segments["client_id"].isin(val_clients)].copy()
    test_df = segments[segments["client_id"].isin(test_clients)].copy()
    if len(train_df) == 0:
        raise RuntimeError("No hay segmentos de entrenamiento para Flower")
    if len(val_df) == 0:
        val_df = train_df
    if len(test_df) == 0:
        test_df = val_df

    transformer = FeatureTransformer.fit(train_df, feature_set=feature_set)
    target_scaler = TargetScaler.fit(train_df[target_column].to_numpy(dtype=np.float32))
    base_model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=True)
    initial_parameters = get_parameters(base_model)
    client_ids = list(train_clients)

    class Client(fl.client.NumPyClient):
        def __init__(self, cid: str):
            self.cid = cid
            self.df = train_df[train_df["client_id"] == cid].copy()
            self.model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=True)

        def get_parameters(self, config):
            return get_parameters(self.model)

        def fit(self, parameters, config):
            set_parameters(self.model, parameters)
            mu = float(config.get("proximal_mu", proximal_mu if strategy_name.lower() == "fedprox" else 0.0))
            loss = _train_local_model(
                self.model,
                self.df,
                transformer,
                target_scaler,
                target_column,
                epochs=int(config.get("local_epochs", local_epochs)),
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                proximal_mu=mu,
                global_parameters=parameters,
            )
            return get_parameters(self.model), len(self.df), {"train_loss": float(loss)}

        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            if len(self.df) == 0:
                return 0.0, 0, {}
            ds = make_dataset(self.df, transformer, target_scaler, target_column)
            loader = DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=False)
            loss_fn = torch.nn.MSELoss()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            self.model.eval()
            losses = []
            with torch.no_grad():
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = self.model(xb)
                    losses.append(float(loss_fn(pred, yb).detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0, len(self.df), {}

    def client_fn(context):
        # Flower >= 1.10 passes a Context object. Older versions passed a string cid.
        # Keep both paths so the project runs on several Flower releases.
        if isinstance(context, str):
            cid_val = context
        else:
            cid_val = None
            try:
                cid_val = str(context.node_config.get("partition-id", 0))
            except Exception:
                pass
            if cid_val is None:
                try:
                    cid_val = str(context.node_config.get("client-id", 0))
                except Exception:
                    cid_val = "0"
        idx = int(cid_val) if str(cid_val).isdigit() else 0
        client_id = client_ids[idx % len(client_ids)]
        c = Client(client_id)
        return c.to_client() if hasattr(c, "to_client") else c

    def evaluate_fn(server_round, parameters, config):
        model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=False)
        set_parameters(model, parameters_to_ndarrays(parameters) if not isinstance(parameters, list) else parameters)
        if len(val_df) == 0:
            return None
        residual_pred = predict_residuals(model, val_df, transformer, target_scaler)
        true = val_df[target_column].to_numpy(dtype=float)
        mse = float(np.mean((residual_pred - true) ** 2)) if len(true) else 0.0
        mae = float(np.mean(np.abs(residual_pred - true))) if len(true) else 0.0
        return mse, {"val_residual_mae_wh": mae}

    class SaveFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.current_parameters = None

        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None:
                self.current_parameters = aggregated[0]
            return aggregated

    StrategyClass = SaveFedAvg
    strategy_kwargs = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min(min_fit_clients, len(client_ids)),
        min_evaluate_clients=min(min_evaluate_clients, len(client_ids)),
        min_available_clients=min(min_available_clients, len(client_ids)),
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "proximal_mu": proximal_mu if strategy_name.lower() == "fedprox" else 0.0},
    )

    if strategy_name.lower() == "fedprox":
        FedProx = getattr(fl.server.strategy, "FedProx", None)
        if FedProx is None:
            raise RuntimeError("La versión instalada de Flower no expone fl.server.strategy.FedProx")

        class SaveFedProx(FedProx):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.current_parameters = None

            def aggregate_fit(self, server_round, results, failures):
                aggregated = super().aggregate_fit(server_round, results, failures)
                if aggregated is not None:
                    self.current_parameters = aggregated[0]
                return aggregated

        StrategyClass = SaveFedProx
        strategy_kwargs["proximal_mu"] = proximal_mu

    strategy = StrategyClass(**strategy_kwargs)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=fl.server.ServerConfig(num_rounds=int(rounds)),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    final_parameters = strategy.current_parameters
    if final_parameters is None:
        final_parameters = ndarrays_to_parameters(initial_parameters)
    final_arrays = parameters_to_ndarrays(final_parameters)
    model = ResidualMLP(transformer.output_dim, hidden_layers=hidden_layers, dropout=dropout, zero_last=False)
    set_parameters(model, final_arrays)
    residual_pred = predict_residuals(model, test_df, transformer, target_scaler)
    segment_pred = test_df[base_prediction_column].to_numpy(dtype=float) + residual_pred
    method = f"M2_flower_{strategy_name.lower()}_residual_mlp"
    route_preds = route_predictions_from_segment_predictions(test_df, segment_pred, method=method)
    summary = summarize_route_predictions(route_preds)

    artifacts = {
        "model": model,
        "transformer": transformer,
        "target_scaler": target_scaler,
        "history": history,
        "n_parameters": count_parameters(model),
        "strategy": strategy_name,
        "train_clients": train_clients,
        "val_clients": val_clients,
        "test_clients": test_clients,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"flower_{strategy_name.lower()}_model.pt")
        transformer.save(out_dir / f"flower_{strategy_name.lower()}_feature_transformer.json")
        target_scaler.save(out_dir / f"flower_{strategy_name.lower()}_target_scaler.json")
        route_preds.to_csv(out_dir / f"flower_{strategy_name.lower()}_route_predictions.csv", index=False)
        summary.to_csv(out_dir / f"flower_{strategy_name.lower()}_summary.csv", index=False)
    return route_preds, summary, artifacts
