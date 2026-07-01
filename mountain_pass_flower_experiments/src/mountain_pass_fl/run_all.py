from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import add_synthetic_clients, build_segments
from .metrics import baseline_route_predictions, save_metrics, summarize_route_predictions
from .plotting import plot_method_mae, plot_prequential, plot_route_scatter
from .prequential import run_prequential_beacon
from .risk import run_risk_analysis
from .train import run_centralized, split_clients


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def dataframe_to_report_table(df: pd.DataFrame) -> str:
    """Return a readable table for REPORT.md without making tabulate mandatory.

    pandas.DataFrame.to_markdown requires the optional dependency `tabulate`.
    If it is not installed, fall back to a compact CSV-style block so the
    experiment never fails at the final report-writing step.
    """
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```text\n" + df.to_csv(index=False) + "```"




def _df_to_markdown_safe(df: pd.DataFrame, *, max_rows: int = 200) -> str:
    """Return a Markdown table without requiring pandas' optional tabulate dependency."""
    if df is None or len(df) == 0:
        return "No hay datos disponibles."
    view = df.head(max_rows).copy()
    # First try pandas' richer Markdown formatter if tabulate is installed.
    try:
        return view.to_markdown(index=False)
    except Exception:
        pass

    def fmt(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value).replace("|", "\|").replace("\n", " ")

    cols = [str(c) for c in view.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in view.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Mostrando {max_rows} de {len(df)} filas._")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    *,
    segments: pd.DataFrame,
    all_route_predictions: pd.DataFrame,
    all_summary: pd.DataFrame,
    risk_summary: pd.DataFrame | None,
    skipped: list[str],
    split_info: dict[str, list[str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Reporte de experimentos: balizas FL para puertos de montaña\n")
    lines.append("## Datos segmentados\n")
    lines.append(f"- Clientes/rutas detectados: **{segments['client_id'].nunique()}**")
    lines.append(f"- Segmentos generados: **{len(segments)}**")
    lines.append(f"- Distancia total segmentada: **{segments['segment_length_m'].sum()/1000:.2f} km**")
    lines.append("\n## Split por clientes\n")
    for key, vals in split_info.items():
        lines.append(f"- {key}: {len(vals)} clientes")
    lines.append("\n## Resumen de error por método\n")
    if all_summary is not None and len(all_summary):
        lines.append(dataframe_to_report_table(all_summary))
    else:
        lines.append("No hay resumen disponible.")
    if risk_summary is not None and len(risk_summary):
        lines.append("\n## Resumen de avisos de riesgo\n")
        lines.append(dataframe_to_report_table(risk_summary))
    if skipped:
        lines.append("\n## Experimentos omitidos o con error\n")
        for s in skipped:
            lines.append(f"- {s}")
    lines.append("\n## Lectura rápida\n")
    lines.append(
        "La comparación principal es de inicio-fin: cada método estima la energía total necesaria antes de entrar al tramo; "
        "la energía real se calcula al salir sumando `energy_used_cum_wh - energy_regen_cum_wh`. "
        "El baseline B1 asume carretera plana y no conoce la orografía. Los modelos M1/M2 aprenden un residual sobre ese baseline usando features topográficas."
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta experimentos EV range + Flower FL sobre CSV de CARLA")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directorio con los CSV")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directorio de resultados")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML de configuración")
    parser.add_argument("--rebuild-segments", action="store_true", help="Reconstruye outputs/segments.csv aunque exista")
    parser.add_argument("--skip-flower", action="store_true", help="Omite los experimentos Flower")
    parser.add_argument("--only-baselines", action="store_true", help="Solo ejecuta segmentación y baselines")
    parser.add_argument("--quick", action="store_true", help="Usa configs/quick.yaml si existe")
    args = parser.parse_args()

    repo_root = Path.cwd()
    config_path = Path(args.config)
    if args.quick and Path("configs/quick.yaml").exists():
        config_path = Path("configs/quick.yaml")
    cfg = load_config(config_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    segment_path = out_dir / "segments.csv"
    if args.rebuild_segments or not segment_path.exists():
        print("[1/7] Construyendo segmentos...")
        segments = build_segments(
            args.data_dir,
            segment_m=float(cfg.get("segment_m", 100.0)),
            min_segment_m=float(cfg.get("min_segment_m", 20.0)),
            expected_speed_kmh=float(cfg.get("expected_speed_kmh", 35.0)),
        )
        syn_cfg = cfg.get("synthetic_clients", {}) or {}
        if syn_cfg.get("enabled", False):
            segments = add_synthetic_clients(
                segments,
                copies_per_client=int(syn_cfg.get("copies_per_client", 0)),
                seed=int(cfg.get("seed", 42)),
                payload_values_kg=syn_cfg.get("payload_kg", [0, 150, 300, 500]),
                eta_factors=syn_cfg.get("eta_factor", [0.96, 1.0, 1.04]),
                crr_factors=syn_cfg.get("crr_factor", [0.95, 1.0, 1.10]),
                capacity_factors=syn_cfg.get("capacity_factor", [0.90, 1.0, 1.05]),
            )
        segments.to_csv(segment_path, index=False)
    else:
        print("[1/7] Leyendo segmentos existentes...")
        segments = pd.read_csv(segment_path)

    print(f"  segmentos={len(segments)} clientes={segments['client_id'].nunique()}")
    print("[2/7] Ejecutando baselines de inicio-fin...")
    baseline_routes = baseline_route_predictions(segments)
    baseline_summary = save_metrics(baseline_routes, out_dir, "baselines")

    clients = sorted(segments["client_id"].unique().tolist())
    split_cfg = cfg.get("split", {}) or {}
    train_clients, val_clients, test_clients = split_clients(
        clients,
        train_fraction=float(split_cfg.get("train_fraction", 0.70)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        test_fraction=float(split_cfg.get("test_fraction", 0.15)),
        seed=int(cfg.get("seed", 42)),
    )
    split_info = {"train": train_clients, "val": val_clients, "test": test_clients}
    (out_dir / "client_split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    all_routes = [baseline_routes]
    skipped: list[str] = []

    if args.only_baselines:
        all_route_predictions = baseline_routes
        all_summary = summarize_route_predictions(all_route_predictions)
        risk_summary = None
        write_report(out_dir, segments=segments, all_route_predictions=all_route_predictions, all_summary=all_summary, risk_summary=risk_summary, skipped=skipped, split_info=split_info)
        return

    train_cfg = cfg.get("training", {}) or {}
    feature_set = cfg.get("feature_set", "entry_topography")
    target_column = cfg.get("target_column", "residual_entry_flat_wh")
    base_prediction_column = cfg.get("base_prediction_column", "baseline_entry_flat_wh")

    print("[3/7] Entrenando modelo centralizado residual MLP...")
    try:
        centralized_routes, centralized_summary, _ = run_centralized(
            segments,
            train_clients=train_clients,
            val_clients=val_clients,
            test_clients=test_clients,
            feature_set=feature_set,
            target_column=target_column,
            base_prediction_column=base_prediction_column,
            hidden_layers=train_cfg.get("hidden_layers", [64, 64, 32]),
            dropout=float(train_cfg.get("dropout", 0.0)),
            batch_size=int(train_cfg.get("batch_size", 32)),
            epochs=int(train_cfg.get("centralized_epochs", 250)),
            lr=float(train_cfg.get("learning_rate", 0.001)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
            patience=int(train_cfg.get("patience", 35)),
            seed=int(cfg.get("seed", 42)),
            out_dir=out_dir / "models",
        )
        all_routes.append(centralized_routes)
    except Exception as exc:
        skipped.append(f"Centralized MLP: {exc}")
        print(f"  ERROR centralizado: {exc}")

    print("[4/7] Ejecutando Flower FedAvg/FedProx...")
    if args.skip_flower:
        skipped.append("Flower omitido por --skip-flower")
    else:
        try:
            from .flower_exp import run_flower
            fl_cfg = cfg.get("flower", {}) or {}
            for strategy in ["fedavg", "fedprox"]:
                try:
                    routes, summary, _ = run_flower(
                        segments,
                        train_clients=train_clients,
                        val_clients=val_clients,
                        test_clients=test_clients,
                        strategy_name=strategy,
                        feature_set=feature_set,
                        target_column=target_column,
                        base_prediction_column=base_prediction_column,
                        hidden_layers=train_cfg.get("hidden_layers", [64, 64, 32]),
                        dropout=float(train_cfg.get("dropout", 0.0)),
                        batch_size=int(train_cfg.get("batch_size", 32)),
                        rounds=int(fl_cfg.get("rounds", 20)),
                        local_epochs=int(train_cfg.get("local_epochs", 5)),
                        lr=float(train_cfg.get("learning_rate", 0.001)),
                        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
                        seed=int(cfg.get("seed", 42)),
                        fraction_fit=float(fl_cfg.get("fraction_fit", 1.0)),
                        fraction_evaluate=float(fl_cfg.get("fraction_evaluate", 1.0)),
                        min_fit_clients=int(fl_cfg.get("min_fit_clients", 2)),
                        min_evaluate_clients=int(fl_cfg.get("min_evaluate_clients", 1)),
                        min_available_clients=int(fl_cfg.get("min_available_clients", 2)),
                        proximal_mu=float(fl_cfg.get("proximal_mu", 0.01)),
                        out_dir=out_dir / "models",
                    )
                    all_routes.append(routes)
                except Exception as exc:
                    skipped.append(f"Flower {strategy}: {exc}")
                    print(f"  ERROR Flower {strategy}: {exc}")
        except Exception as exc:
            skipped.append(f"Flower import/run: {exc}")
            print(f"  ERROR Flower general: {exc}")

    print("[5/7] Ejecutando evaluación prequential de baliza...")
    try:
        preq_routes, preq_summary = run_prequential_beacon(
            segments,
            feature_set=feature_set,
            target_column=target_column,
            base_prediction_column=base_prediction_column,
            hidden_layers=train_cfg.get("hidden_layers", [64, 64, 32]),
            dropout=float(train_cfg.get("dropout", 0.0)),
            batch_size=int(train_cfg.get("batch_size", 32)),
            local_epochs=int(train_cfg.get("local_epochs", 5)),
            lr=float(train_cfg.get("learning_rate", 0.001)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
            seed=int(cfg.get("seed", 42)),
            out_dir=out_dir / "prequential",
        )
        all_routes.append(preq_routes)
    except Exception as exc:
        skipped.append(f"Prequential: {exc}")
        print(f"  ERROR prequential: {exc}")

    print("[6/7] Agregando métricas y análisis de riesgo...")
    all_route_predictions = pd.concat([x for x in all_routes if x is not None and len(x)], ignore_index=True)
    all_route_predictions.to_csv(out_dir / "all_route_predictions.csv", index=False)
    all_summary = summarize_route_predictions(all_route_predictions)
    all_summary.to_csv(out_dir / "all_summary.csv", index=False)

    risk_summary = None
    try:
        risk_cfg = cfg.get("risk", {}) or {}
        _, risk_summary = run_risk_analysis(
            all_route_predictions,
            out_dir / "risk",
            soc_grid_percent=[float(x) for x in risk_cfg.get("soc_grid_percent", [10, 15, 20, 25, 30])],
            reserve_soc_percent=float(risk_cfg.get("reserve_soc_percent", 5.0)),
        )
    except Exception as exc:
        skipped.append(f"Risk analysis: {exc}")
        print(f"  ERROR riesgo: {exc}")

    (out_dir / "skipped_experiments.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    print("[7/7] Generando figuras y reporte...")
    try:
        plot_dir = out_dir / "plots"
        plot_route_scatter(all_route_predictions, plot_dir)
        plot_method_mae(all_summary, plot_dir)
        if 'preq_routes' in locals():
            plot_prequential(preq_routes, plot_dir)
    except Exception as exc:
        skipped.append(f"Plots: {exc}")
        print(f"  ERROR plots: {exc}")

    write_report(
        out_dir,
        segments=segments,
        all_route_predictions=all_route_predictions,
        all_summary=all_summary,
        risk_summary=risk_summary,
        skipped=skipped,
        split_info=split_info,
    )
    print(f"Listo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
