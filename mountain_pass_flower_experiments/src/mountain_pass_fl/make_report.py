from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .run_all import write_report


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenera REPORT.md a partir de outputs ya existentes")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directorio de resultados")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    segments = _read_csv_if_exists(out_dir / "segments.csv")
    all_routes = _read_csv_if_exists(out_dir / "all_route_predictions.csv")
    all_summary = _read_csv_if_exists(out_dir / "all_summary.csv")
    risk_summary = _read_csv_if_exists(out_dir / "risk" / "risk_summary.csv")

    if segments is None:
        raise FileNotFoundError(f"No existe {out_dir / 'segments.csv'}")
    if all_routes is None:
        raise FileNotFoundError(f"No existe {out_dir / 'all_route_predictions.csv'}")
    if all_summary is None:
        all_summary = pd.DataFrame()

    split_path = out_dir / "client_split.json"
    if split_path.exists():
        split_info = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split_info = {"train": [], "val": [], "test": []}

    skipped_path = out_dir / "skipped_experiments.json"
    if skipped_path.exists():
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
    else:
        skipped = []

    write_report(
        out_dir,
        segments=segments,
        all_route_predictions=all_routes,
        all_summary=all_summary,
        risk_summary=risk_summary,
        skipped=skipped,
        split_info=split_info,
    )
    print(f"Reporte regenerado: {(out_dir / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
