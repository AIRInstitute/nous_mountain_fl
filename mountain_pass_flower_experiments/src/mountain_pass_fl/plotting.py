from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_route_scatter(route_preds: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if route_preds.empty:
        return
    for method, g in route_preds.groupby("method"):
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(g["true_wh"] / 1000.0, g["pred_wh"] / 1000.0)
        lim_min = min(g["true_wh"].min(), g["pred_wh"].min()) / 1000.0
        lim_max = max(g["true_wh"].max(), g["pred_wh"].max()) / 1000.0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
        plt.xlabel("Energía real del puerto/tramo [kWh]")
        plt.ylabel("Energía predicha [kWh]")
        plt.title(method)
        plt.tight_layout()
        safe = method.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"scatter_{safe}.png", dpi=160)
        plt.close(fig)


def plot_method_mae(summary: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty or "mae_kwh" not in summary.columns:
        return
    s = summary.sort_values("mae_kwh")
    fig = plt.figure(figsize=(8, max(4, 0.45 * len(s))))
    plt.barh(s["method"], s["mae_kwh"])
    plt.xlabel("MAE energía total [kWh]")
    plt.ylabel("Método")
    plt.tight_layout()
    fig.savefig(out_dir / "mae_by_method.png", dpi=160)
    plt.close(fig)


def plot_prequential(preq: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if preq.empty or "arrival_order" not in preq.columns:
        return
    fig = plt.figure(figsize=(7, 4))
    g = preq.sort_values("arrival_order")
    plt.plot(g["arrival_order"], g["abs_error_wh"] / 1000.0, marker="o")
    plt.xlabel("Número de cruce / orden de llegada")
    plt.ylabel("Error absoluto de energía [kWh]")
    plt.title("Evaluación prequential: predecir antes, actualizar al salir")
    plt.tight_layout()
    fig.savefig(out_dir / "prequential_abs_error.png", dpi=160)
    plt.close(fig)


def plot_cv_mae_boxplot(fold_summary: pd.DataFrame, out_dir: str | Path, *, protocol: str | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fold_summary.empty or "mae_kwh" not in fold_summary.columns:
        return
    df = fold_summary.copy()
    if protocol is not None:
        df = df[df["protocol"] == protocol]
    if df.empty:
        return
    methods = df.groupby("method")["mae_kwh"].median().sort_values().index.tolist()
    data = [df.loc[df["method"] == m, "mae_kwh"].dropna().to_numpy() for m in methods]
    if not data:
        return
    fig = plt.figure(figsize=(max(8, 0.55 * len(methods)), 5))
    plt.boxplot(data, tick_labels=methods, showfliers=True)
    plt.ylabel("MAE energía total por fold [kWh]")
    title = "Validación cruzada"
    if protocol:
        title += f" - {protocol}"
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    suffix = protocol if protocol else "all_protocols"
    fig.savefig(out_dir / f"cv_mae_boxplot_{suffix}.png", dpi=160)
    plt.close(fig)


def plot_learning_curve_mae(learning_summary: pd.DataFrame, out_dir: str | Path, *, eval_mode: str = "remaining_unseen") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if learning_summary.empty or "mae_kwh_mean" not in learning_summary.columns:
        return
    df = learning_summary[learning_summary["eval_mode"] == eval_mode].copy() if "eval_mode" in learning_summary.columns else learning_summary.copy()
    if df.empty:
        return
    # Keep the plot readable: draw one line per method/batch/strategy combination.
    fig = plt.figure(figsize=(9, 5))
    for keys, g in df.groupby(["method", "batch_size"], dropna=False):
        method, batch_size = keys
        g = g.sort_values("n_seen_clients")
        label = f"{method} | batch={batch_size}"
        plt.plot(g["n_seen_clients"], g["mae_kwh_mean"], marker="o", label=label)
    plt.xlabel("Coches previos vistos por la baliza")
    plt.ylabel("MAE energía total [kWh]")
    plt.title(f"Curva de aprendizaje de la baliza ({eval_mode})")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / f"learning_curve_mae_{eval_mode}.png", dpi=160)
    plt.close(fig)


def plot_learning_curve_false_safe(risk_summary: pd.DataFrame, out_dir: str | Path, *, eval_mode: str = "remaining_unseen") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if risk_summary.empty or "false_safe_rate" not in risk_summary.columns:
        return
    df = risk_summary[risk_summary["eval_mode"] == eval_mode].copy() if "eval_mode" in risk_summary.columns else risk_summary.copy()
    if df.empty:
        return
    fig = plt.figure(figsize=(9, 5))
    for keys, g in df.groupby(["method", "batch_size"], dropna=False):
        method, batch_size = keys
        g = g.sort_values("n_seen_clients")
        label = f"{method} | batch={batch_size}"
        plt.plot(g["n_seen_clients"], g["false_safe_rate"], marker="o", label=label)
    plt.xlabel("Coches previos vistos por la baliza")
    plt.ylabel("False-safe rate")
    plt.title(f"Riesgo: falsos 'puedes cruzar' ({eval_mode})")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / f"learning_curve_false_safe_{eval_mode}.png", dpi=160)
    plt.close(fig)
