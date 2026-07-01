from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .cross_validation import client_metadata, filter_segments_by_min_route_distance, run_cross_validation
from .data import add_synthetic_clients, build_segments
from .learning_curve import run_beacon_learning_curve, run_flower_learning_checkpoints
from .plotting import plot_cv_mae_boxplot, plot_learning_curve_false_safe, plot_learning_curve_mae
from .risk import make_risk_table, summarize_risk_by
from .run_all import dataframe_to_report_table, load_config


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _build_or_load_segments(args, cfg: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    segment_path = out_dir / "segments.csv"
    if args.rebuild_segments or not segment_path.exists():
        print("[1/6] Construyendo segmentos...")
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
        min_route_distance_m = _get(cfg, "data_filters.min_route_distance_m", None)
        if min_route_distance_m:
            before = segments["client_id"].nunique()
            segments = filter_segments_by_min_route_distance(segments, float(min_route_distance_m))
            after = segments["client_id"].nunique()
            print(f"  filtro min_route_distance_m={min_route_distance_m}: clientes {before} -> {after}")
        segments.to_csv(segment_path, index=False)
    else:
        print("[1/6] Leyendo segmentos existentes...")
        segments = pd.read_csv(segment_path)
    meta = client_metadata(segments)
    meta.to_csv(out_dir / "client_metadata.csv", index=False)
    print(f"  segmentos={len(segments)} clientes={segments['client_id'].nunique()} distancia_total={segments['segment_length_m'].sum()/1000:.2f} km")
    return segments


def _save_grouped_risk(
    route_predictions: pd.DataFrame,
    out_dir: Path,
    *,
    soc_grid_percent: list[float],
    reserve_soc_percent: float,
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_table = make_risk_table(route_predictions, soc_grid_percent=soc_grid_percent, reserve_soc_percent=reserve_soc_percent)
    risk_summary = summarize_risk_by(risk_table, group_cols=group_cols)
    risk_table.to_csv(out_dir / "risk_decisions.csv", index=False)
    risk_summary.to_csv(out_dir / "risk_summary_grouped.csv", index=False)
    return risk_table, risk_summary


def write_extended_report(
    out_dir: Path,
    *,
    segments: pd.DataFrame,
    cv_summary: pd.DataFrame | None,
    lc_summary: pd.DataFrame | None,
    flower_lc_summary: pd.DataFrame | None,
    cv_risk_summary: pd.DataFrame | None,
    lc_risk_summary: pd.DataFrame | None,
    skipped: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Reporte extendido: validación cruzada y curva de aprendizaje de baliza\n")
    lines.append("## Datos\n")
    lines.append(f"- Clientes/rutas: **{segments['client_id'].nunique()}**")
    lines.append(f"- Segmentos: **{len(segments)}**")
    lines.append(f"- Distancia total segmentada: **{segments['segment_length_m'].sum()/1000:.2f} km**")
    lines.append("\n## Qué mide este reporte\n")
    lines.append(
        "La métrica principal sigue siendo de inicio-fin: energía estimada antes de entrar al puerto frente a energía real al salir. "
        "La validación cruzada se hace por clientes/rutas completas, nunca por filas ni por segmentos aleatorios."
    )
    if cv_summary is not None and len(cv_summary):
        lines.append("\n## Validación cruzada: resumen agregado por protocolo\n")
        # Keep the report readable by showing the most important columns.
        cols = [c for c in ["protocol", "method", "n_folds", "total_test_routes", "mae_kwh_mean", "mae_kwh_std", "bias_wh_mean", "mape_percent_mean", "mae_soc_final_percent_mean"] if c in cv_summary.columns]
        lines.append(dataframe_to_report_table(cv_summary[cols].copy()))
    if cv_risk_summary is not None and len(cv_risk_summary):
        lines.append("\n## Validación cruzada: riesgo por protocolo\n")
        cols = [c for c in ["protocol", "method", "false_safe_rate", "false_warning_rate", "risk_recall", "n_decisions"] if c in cv_risk_summary.columns]
        lines.append(dataframe_to_report_table(cv_risk_summary[cols].copy()))
    if lc_summary is not None and len(lc_summary):
        lines.append("\n## Curva de aprendizaje de la baliza\n")
        lines.append(
            "`remaining_unseen` evalúa el modelo tras ver k coches sobre los coches que aún no han pasado. "
            "`all_routes_replay` es diagnóstico: tras ver k coches, reevalúa todas las rutas, por lo que no debe leerse como generalización pura cuando k > 0."
        )
        cols = [c for c in ["eval_mode", "batch_size", "n_seen_clients", "method", "n_permutations", "mae_kwh_mean", "mae_kwh_std", "bias_wh_mean", "mape_percent_mean"] if c in lc_summary.columns]
        show = lc_summary[cols].copy()
        # Show a compact subset: remaining_unseen primarily.
        if "eval_mode" in show.columns:
            show = show[show["eval_mode"] == "remaining_unseen"].head(80)
        lines.append(dataframe_to_report_table(show))
    if flower_lc_summary is not None and len(flower_lc_summary):
        lines.append("\n## Checkpoints Flower explícitos\n")
        cols = [c for c in ["eval_mode", "n_seen_clients", "method", "strategy", "mae_kwh_mean", "bias_wh_mean", "mape_percent_mean"] if c in flower_lc_summary.columns]
        lines.append(dataframe_to_report_table(flower_lc_summary[cols].head(80).copy()))
    if lc_risk_summary is not None and len(lc_risk_summary):
        lines.append("\n## Curva de aprendizaje: falsos 'puedes cruzar'\n")
        cols = [c for c in ["eval_mode", "batch_size", "n_seen_clients", "method", "false_safe_rate", "false_warning_rate", "risk_recall", "n_decisions"] if c in lc_risk_summary.columns]
        show = lc_risk_summary[cols].copy()
        if "eval_mode" in show.columns:
            show = show[show["eval_mode"] == "remaining_unseen"].head(100)
        lines.append(dataframe_to_report_table(show))
    if skipped:
        lines.append("\n## Experimentos omitidos o con error\n")
        for item in skipped:
            lines.append(f"- {item}")
    lines.append("\n## Interpretación esperada\n")
    lines.append(
        "Busca especialmente dos efectos: (1) que los métodos sin orografía mantengan bias negativo, es decir, infraestimen energía; "
        "y (2) que la curva de baliza reduzca MAE y false-safe rate conforme aumenta `n_seen_clients`."
    )
    (out_dir / "EXTENDED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos extendidos: CV + curva de aprendizaje de baliza")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="outputs_extended")
    parser.add_argument("--config", type=str, default="configs/extended.yaml")
    parser.add_argument("--rebuild-segments", action="store_true")
    parser.add_argument("--skip-flower", action="store_true", help="Omitir los checkpoints Flower y cualquier Flower de CV")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-learning-curve", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Usa configs/quick_extended.yaml si existe")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if args.quick and Path("configs/quick_extended.yaml").exists():
        cfg_path = Path("configs/quick_extended.yaml")
    cfg = load_config(cfg_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    segments = _build_or_load_segments(args, cfg, out_dir)

    seed = int(cfg.get("seed", 42))
    train_cfg = cfg.get("training", {}) or {}
    fl_cfg = cfg.get("flower", {}) or {}
    feature_set = cfg.get("feature_set", "entry_topography")
    target_column = cfg.get("target_column", "residual_entry_flat_wh")
    base_prediction_column = cfg.get("base_prediction_column", "baseline_entry_flat_wh")
    risk_cfg = cfg.get("risk", {}) or {}
    soc_grid = [float(x) for x in risk_cfg.get("soc_grid_percent", [8, 10, 12, 15, 20, 25, 30, 40])]
    reserve_soc = float(risk_cfg.get("reserve_soc_percent", 5.0))

    skipped_all: list[str] = []
    cv_summary = cv_routes = cv_fold_summary = None
    cv_risk_summary = None
    lc_summary = lc_routes = lc_point_summary = None
    flower_lc_summary = flower_lc_routes = None
    lc_risk_summary = None

    if not args.skip_cv and bool(_get(cfg, "cross_validation.enabled", True)):
        print("[2/6] Ejecutando validación cruzada por rutas/clientes...")
        cv_cfg = cfg.get("cross_validation", {}) or {}
        cv_routes, cv_fold_summary, cv_summary, skipped = run_cross_validation(
            segments,
            cfg=cv_cfg,
            seed=seed,
            feature_set=feature_set,
            target_column=target_column,
            base_prediction_column=base_prediction_column,
            hidden_layers=train_cfg.get("hidden_layers", [64, 64, 32]),
            dropout=float(train_cfg.get("dropout", 0.0)),
            batch_size=int(train_cfg.get("batch_size", 32)),
            centralized_epochs=int(train_cfg.get("cv_centralized_epochs", train_cfg.get("centralized_epochs", 160))),
            local_epochs=int(train_cfg.get("cv_local_epochs", train_cfg.get("local_epochs", 3))),
            lr=float(train_cfg.get("learning_rate", 0.001)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
            patience=int(train_cfg.get("cv_patience", train_cfg.get("patience", 25))),
            run_centralized_models=bool(cv_cfg.get("run_centralized", True)),
            run_flower_models=bool(cv_cfg.get("run_flower", False)) and not args.skip_flower,
            flower_strategies=cv_cfg.get("flower_strategies", ["fedavg"]),
            flower_rounds=int(cv_cfg.get("flower_rounds", fl_cfg.get("rounds", 8))),
            flower_fraction_fit=float(fl_cfg.get("fraction_fit", 1.0)),
            flower_fraction_evaluate=float(fl_cfg.get("fraction_evaluate", 1.0)),
            flower_min_fit_clients=int(fl_cfg.get("min_fit_clients", 2)),
            flower_min_evaluate_clients=int(fl_cfg.get("min_evaluate_clients", 1)),
            flower_min_available_clients=int(fl_cfg.get("min_available_clients", 2)),
            flower_proximal_mu=float(fl_cfg.get("proximal_mu", 0.01)),
            out_dir=out_dir / "cv",
        )
        skipped_all.extend([f"CV: {x}" for x in skipped])
        if cv_routes is not None and len(cv_routes):
            _, cv_risk_summary = _save_grouped_risk(
                cv_routes,
                out_dir / "cv" / "risk",
                soc_grid_percent=soc_grid,
                reserve_soc_percent=reserve_soc,
                group_cols=["protocol", "method"],
            )
    else:
        print("[2/6] Validación cruzada omitida")

    if not args.skip_learning_curve and bool(_get(cfg, "learning_curve.enabled", True)):
        print("[3/6] Ejecutando curva de aprendizaje de baliza...")
        lc_cfg = cfg.get("learning_curve", {}) or {}
        lc_routes, lc_point_summary, lc_summary, skipped = run_beacon_learning_curve(
            segments,
            cfg=lc_cfg,
            seed=seed,
            feature_set=feature_set,
            target_column=target_column,
            base_prediction_column=base_prediction_column,
            hidden_layers=train_cfg.get("hidden_layers", [64, 64, 32]),
            dropout=float(train_cfg.get("dropout", 0.0)),
            batch_size_train=int(train_cfg.get("batch_size", 32)),
            local_epochs=int(train_cfg.get("learning_curve_local_epochs", train_cfg.get("local_epochs", 5))),
            lr=float(train_cfg.get("learning_rate", 0.001)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
            proximal_mu=float(fl_cfg.get("proximal_mu", 0.01)),
            out_dir=out_dir / "learning_curve",
        )
        skipped_all.extend([f"Learning curve: {x}" for x in skipped])
        if lc_routes is not None and len(lc_routes):
            _, lc_risk_summary = _save_grouped_risk(
                lc_routes,
                out_dir / "learning_curve" / "risk",
                soc_grid_percent=soc_grid,
                reserve_soc_percent=reserve_soc,
                group_cols=["eval_mode", "batch_size", "n_seen_clients", "method"],
            )

        if not args.skip_flower:
            print("[4/6] Ejecutando checkpoints Flower explícitos de la curva...")
            flower_ckpt_cfg = lc_cfg.get("flower_checkpoints", {}) or {}
            flower_lc_routes, flower_lc_point, flower_lc_summary, skipped = run_flower_learning_checkpoints(
                segments,
                cfg=flower_ckpt_cfg,
                seed=seed,
                feature_set=feature_set,
                target_column=target_column,
                base_prediction_column=base_prediction_column,
                hidden_layers=train_cfg.get("hidden_layers", [64, 64, 32]),
                dropout=float(train_cfg.get("dropout", 0.0)),
                batch_size_train=int(train_cfg.get("batch_size", 32)),
                local_epochs=int(train_cfg.get("flower_checkpoint_local_epochs", train_cfg.get("local_epochs", 3))),
                lr=float(train_cfg.get("learning_rate", 0.001)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
                proximal_mu=float(fl_cfg.get("proximal_mu", 0.01)),
                out_dir=out_dir / "learning_curve" / "flower_checkpoints",
            )
            skipped_all.extend([f"Flower checkpoints: {x}" for x in skipped])
        else:
            print("[4/6] Checkpoints Flower omitidos por --skip-flower")
    else:
        print("[3/6] Curva de aprendizaje omitida")
        print("[4/6] Checkpoints Flower omitidos")

    print("[5/6] Generando figuras...")
    plot_dir = out_dir / "plots"
    if cv_fold_summary is not None and len(cv_fold_summary):
        plot_cv_mae_boxplot(cv_fold_summary, plot_dir)
        for protocol in sorted(cv_fold_summary["protocol"].dropna().unique().tolist()):
            plot_cv_mae_boxplot(cv_fold_summary, plot_dir, protocol=str(protocol))
    if lc_summary is not None and len(lc_summary):
        plot_learning_curve_mae(lc_summary, plot_dir, eval_mode="remaining_unseen")
        plot_learning_curve_mae(lc_summary, plot_dir, eval_mode="all_routes_replay")
    if lc_risk_summary is not None and len(lc_risk_summary):
        plot_learning_curve_false_safe(lc_risk_summary, plot_dir, eval_mode="remaining_unseen")
        plot_learning_curve_false_safe(lc_risk_summary, plot_dir, eval_mode="all_routes_replay")

    print("[6/6] Escribiendo reporte extendido...")
    pd.DataFrame({"skipped": skipped_all}).to_csv(out_dir / "extended_skipped.csv", index=False)
    write_extended_report(
        out_dir,
        segments=segments,
        cv_summary=cv_summary,
        lc_summary=lc_summary,
        flower_lc_summary=flower_lc_summary,
        cv_risk_summary=cv_risk_summary,
        lc_risk_summary=lc_risk_summary,
        skipped=skipped_all,
    )
    print(f"Listo. Resultados extendidos en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
