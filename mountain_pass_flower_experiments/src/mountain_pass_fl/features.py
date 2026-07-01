from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ENTRY_TOPO_NUMERIC = [
    "segment_id_norm",
    "segment_length_m",
    "distance_from_start_m",
    "route_distance_m",
    "altitude_start_m",
    "delta_z_m",
    "uphill_m",
    "downhill_m",
    "grade_from_z_deg",
    "grade_mean_deg",
    "grade_max_deg",
    "grade_min_deg",
    "curvature_abs_mean",
    "ambient_temp_c",
    "precip_intensity",
    "wind_long_ms",
    "wind_speed_ms",
    "soc_start_percent",
    "capacity_usable_wh",
    "battery_temp_c",
    "expected_speed_kmh",
    "baseline_entry_flat_wh",
    "baseline_nominal_wh",
]

ONLINE_NUMERIC = ENTRY_TOPO_NUMERIC + [
    "speed_mean_kmh",
    "speed_max_kmh",
    "acc_long_mean",
    "acc_long_abs_mean",
    "acc_lat_abs_mean",
    "throttle_mean",
    "brake_mean",
    "steering_abs_mean",
]

CATEGORICAL = ["vehicle_key", "road_condition"]


@dataclass
class FeatureTransformer:
    numeric_columns: list[str]
    categorical_columns: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    categories: dict[str, list[str]]

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_set: str = "entry_topography") -> "FeatureTransformer":
        numeric_columns = get_numeric_columns(feature_set)
        categorical_columns = CATEGORICAL.copy()
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for col in numeric_columns:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([0.0])
            mean = float(vals.mean()) if vals.notna().any() else 0.0
            std = float(vals.std(ddof=0)) if vals.notna().any() else 1.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            means[col] = mean
            stds[col] = std
        categories: dict[str, list[str]] = {}
        for col in categorical_columns:
            if col in df.columns:
                cats = sorted([str(x) for x in df[col].dropna().unique().tolist()])
            else:
                cats = ["unknown"]
            if not cats:
                cats = ["unknown"]
            categories[col] = cats
        return cls(numeric_columns, categorical_columns, means, stds, categories)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for col in self.numeric_columns:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(self.means[col]).to_numpy(dtype=np.float32)
            else:
                vals = np.full(len(df), self.means[col], dtype=np.float32)
            blocks.append(((vals - self.means[col]) / self.stds[col]).reshape(-1, 1).astype(np.float32))
        for col in self.categorical_columns:
            cats = self.categories.get(col, ["unknown"])
            values = df[col].astype(str).fillna("unknown").tolist() if col in df.columns else ["unknown"] * len(df)
            mat = np.zeros((len(df), len(cats)), dtype=np.float32)
            idx = {c: i for i, c in enumerate(cats)}
            for r, val in enumerate(values):
                if val in idx:
                    mat[r, idx[val]] = 1.0
            blocks.append(mat)
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return len(self.numeric_columns) + sum(len(v) for v in self.categories.values())

    @property
    def output_names(self) -> list[str]:
        names = [f"num::{c}" for c in self.numeric_columns]
        for col, cats in self.categories.items():
            names.extend([f"cat::{col}={cat}" for cat in cats])
        return names

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTransformer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def get_numeric_columns(feature_set: str) -> list[str]:
    if feature_set == "entry_topography":
        return ENTRY_TOPO_NUMERIC.copy()
    if feature_set == "online_kinematics":
        return ONLINE_NUMERIC.copy()
    raise ValueError(f"feature_set desconocido: {feature_set}")


@dataclass
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, y: Iterable[float]) -> "TargetScaler":
        arr = np.asarray(list(y), dtype=np.float32)
        # Residual models are initialized with a zero last layer. Keeping the
        # target mean at zero guarantees that an untrained model predicts a
        # zero residual, i.e., exactly the physical baseline. This avoids
        # leaking the average residual of future vehicles into the cold-start
        # beacon/prequential experiment.
        mean = 0.0
        std = float(np.nanstd(arr)) if arr.size else 1.0
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, y: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(y), dtype=np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, y_scaled: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(y_scaled), dtype=np.float32)
        return (arr * self.std + self.mean).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean, "std": self.std}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TargetScaler":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
