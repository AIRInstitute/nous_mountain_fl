from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

G = 9.80665
RHO_AIR = 1.225


@dataclass(frozen=True)
class EVParams:
    vehicle_key: str
    display_name: str
    mass_kg: float
    cd: float
    frontal_area_m2: float
    c_rr: float
    usable_wh: float
    eta_drive: float
    eta_regen: float
    aux_power_base_w: float
    nominal_wh_per_km: float
    max_regen_power_w: float = 80_000.0


EV_LIBRARY: dict[str, EVParams] = {
    "tesla_model3": EVParams(
        vehicle_key="tesla_model3",
        display_name="Tesla Model 3",
        mass_kg=1611.0,
        cd=0.23,
        frontal_area_m2=2.22,
        c_rr=0.011,
        usable_wh=72_000.0,
        eta_drive=0.90,
        eta_regen=0.70,
        aux_power_base_w=500.0,
        nominal_wh_per_km=160.0,
        max_regen_power_w=75_000.0,
    ),
    "audi_etron": EVParams(
        vehicle_key="audi_etron",
        display_name="Audi e-tron",
        mass_kg=2565.0,
        cd=0.28,
        frontal_area_m2=2.65,
        c_rr=0.012,
        usable_wh=86_400.0,
        eta_drive=0.88,
        eta_regen=0.68,
        aux_power_base_w=500.0,
        nominal_wh_per_km=280.0,
        max_regen_power_w=100_000.0,
    ),
    "cybertruck": EVParams(
        vehicle_key="cybertruck",
        display_name="Tesla Cybertruck",
        mass_kg=3000.0,
        cd=0.34,
        frontal_area_m2=3.20,
        c_rr=0.013,
        usable_wh=120_000.0,
        eta_drive=0.87,
        eta_regen=0.65,
        aux_power_base_w=500.0,
        nominal_wh_per_km=330.0,
        max_regen_power_w=120_000.0,
    ),
}


def standardize_vehicle_key(value: Any | None, fallback_text: str | None = None) -> str:
    """Map scenario/file strings to normalized vehicle keys."""
    text = ""
    if value is not None and not (isinstance(value, float) and np.isnan(value)):
        text += str(value).lower() + " "
    if fallback_text:
        text += str(fallback_text).lower()

    if "audi" in text or "etron" in text or "e-tron" in text:
        return "audi_etron"
    if "cyber" in text:
        return "cybertruck"
    if "tesla" in text or "model3" in text or "model_3" in text:
        return "tesla_model3"
    return "tesla_model3"


def infer_road_condition(text: Any | None) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "dry"
    t = str(text).lower()
    if "snow" in t or "neg5" in t or "-5" in t:
        return "snow"
    if "wet" in t or "rain" in t or "lluv" in t:
        return "wet"
    if "ice" in t:
        return "ice"
    return "dry"


def infer_start_soc_from_text(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"soc\s*([0-9]{1,3})", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def infer_temperature_from_text(text: str | None) -> float | None:
    if not text:
        return None
    t = text.lower()
    if "neg5" in t or "-5" in t or "snow" in t:
        return -5.0
    m = re.search(r"(?:dry|wet|rain|snow)[_\- ]*([0-9]{1,2})", t)
    if m:
        return float(m.group(1))
    if "24c" in t or "24_c" in t:
        return 24.0
    if "10c" in t or "10_c" in t or "wet10" in t:
        return 10.0
    return None


def hvac_aux_power(t_ambient_c: float | np.ndarray) -> np.ndarray:
    """Simple auxiliary/HVAC power model in W.

    It is intentionally basic and non-learned. It roughly follows the model used in
    the CARLA scripts: mild weather has a small fixed auxiliary load, cold weather
    adds heating demand.
    """
    t = np.asarray(t_ambient_c, dtype=float)
    out = np.full_like(t, 500.0, dtype=float)
    cold = t < 20.0
    out[cold] = 500.0 + (20.0 - t[cold]) * 100.0
    very_cold = t < 0.0
    out[very_cold] = 2500.0 + (0.0 - t[very_cold]) * 250.0
    return out


def battery_capacity_factor(t_ambient_c: float | np.ndarray) -> np.ndarray:
    t = np.asarray(t_ambient_c, dtype=float)
    factor = np.ones_like(t, dtype=float)
    mid = (t < 20.0) & (t >= 0.0)
    factor[mid] = 1.0 - (20.0 - t[mid]) * 0.005
    low = t < 0.0
    factor[low] = 0.90 - np.minimum(0.15, (0.0 - t[low]) * 0.0075)
    return np.clip(factor, 0.70, 1.05)


def rolling_multiplier(condition: Any) -> float:
    c = infer_road_condition(condition)
    if c == "wet":
        return 1.20
    if c == "snow":
        return 2.00
    if c == "ice":
        return 1.50
    return 1.00


def _as_array(values: Any, n: int, default: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=float)
    arr = np.asarray(values)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    arr = arr.astype(float)
    if len(arr) != n:
        return np.resize(arr, n).astype(float)
    return arr


def physics_energy_from_profile(
    *,
    speed_kmh: Any,
    acc_long_ms2: Any | None,
    grade_deg: Any | None,
    dt_s: Any,
    vehicle_key: str,
    road_condition: Any | None = None,
    ambient_temp_c: Any | None = None,
    wind_long_ms: Any | None = None,
    grade_zero: bool = False,
    use_weather: bool = True,
    use_recorded_aux_power: Any | None = None,
    allow_regen: bool = True,
) -> dict[str, float]:
    """Integrate a longitudinal physical road-load model over a profile.

    The function returns net battery energy as well as separated used/regen energy.
    It deliberately remains a basic physics model: no neural network, no fitted
    parameters, no access to cumulative energy labels.
    """
    speed_kmh_arr = np.asarray(speed_kmh, dtype=float)
    n = len(speed_kmh_arr)
    params = EV_LIBRARY[standardize_vehicle_key(vehicle_key)]

    v = np.maximum(speed_kmh_arr / 3.6, 0.0)
    dt = np.maximum(_as_array(dt_s, n, default=0.05), 1e-6)
    acc = _as_array(acc_long_ms2, n, default=0.0)
    if grade_zero:
        grade_rad = np.zeros(n, dtype=float)
    else:
        grade_rad = np.deg2rad(_as_array(grade_deg, n, default=0.0))

    wind = _as_array(wind_long_ms, n, default=0.0)
    v_rel = np.maximum(v - wind, 0.0)

    if use_weather:
        if road_condition is None:
            mult = np.ones(n, dtype=float)
        elif isinstance(road_condition, pd.Series) or isinstance(road_condition, list) or isinstance(road_condition, np.ndarray):
            mult = np.array([rolling_multiplier(x) for x in list(road_condition)], dtype=float)
        else:
            mult = np.full(n, rolling_multiplier(road_condition), dtype=float)
    else:
        mult = np.ones(n, dtype=float)
    crr = params.c_rr * mult

    if use_recorded_aux_power is not None:
        aux = _as_array(use_recorded_aux_power, n, default=params.aux_power_base_w)
    elif use_weather:
        temp = _as_array(ambient_temp_c, n, default=24.0)
        aux = hvac_aux_power(temp)
    else:
        aux = np.full(n, params.aux_power_base_w, dtype=float)

    f_roll = crr * params.mass_kg * G * np.cos(grade_rad)
    f_aero = 0.5 * RHO_AIR * params.cd * params.frontal_area_m2 * v_rel**2
    f_grade = params.mass_kg * G * np.sin(grade_rad)
    f_inertia = params.mass_kg * acc
    f_total = f_roll + f_aero + f_grade + f_inertia
    p_wheels = f_total * v

    p_batt = np.empty(n, dtype=float)
    traction = p_wheels >= 0
    p_batt[traction] = p_wheels[traction] / params.eta_drive + aux[traction]
    if allow_regen:
        regen_power = p_wheels[~traction] * params.eta_regen  # negative
        regen_power = np.maximum(regen_power, -params.max_regen_power_w)
        p_batt[~traction] = regen_power + aux[~traction]
    else:
        p_batt[~traction] = aux[~traction]

    e_wh = p_batt * dt / 3600.0
    used_wh = float(np.sum(np.maximum(e_wh, 0.0)))
    regen_wh = float(np.sum(np.maximum(-e_wh, 0.0)))
    net_wh = float(np.sum(e_wh))
    return {"net_wh": net_wh, "used_wh": used_wh, "regen_wh": regen_wh}


def constant_speed_flat_energy(
    *,
    distance_m: float,
    vehicle_key: str,
    expected_speed_kmh: float = 35.0,
    ambient_temp_c: float = 24.0,
    road_condition: str = "dry",
    use_weather: bool = False,
) -> float:
    """Entry-time physical baseline with no topography and no future trajectory.

    The only required route information is distance. It assumes flat road, constant
    expected speed, no acceleration, and nominal vehicle parameters. This is the
    primary topography-agnostic baseline.
    """
    if distance_m <= 0:
        return 0.0
    speed_kmh = max(float(expected_speed_kmh), 1.0)
    duration_s = distance_m / (speed_kmh / 3.6)
    return physics_energy_from_profile(
        speed_kmh=[speed_kmh],
        acc_long_ms2=[0.0],
        grade_deg=[0.0],
        dt_s=[duration_s],
        vehicle_key=vehicle_key,
        road_condition=[road_condition],
        ambient_temp_c=[ambient_temp_c],
        wind_long_ms=[0.0],
        grade_zero=True,
        use_weather=use_weather,
        allow_regen=False,
    )["net_wh"]


def nominal_energy(distance_m: float, vehicle_key: str) -> float:
    params = EV_LIBRARY[standardize_vehicle_key(vehicle_key)]
    return float(distance_m / 1000.0 * params.nominal_wh_per_km)


def capacity_wh_for_vehicle(vehicle_key: str, ambient_temp_c: float | None = None) -> float:
    params = EV_LIBRARY[standardize_vehicle_key(vehicle_key)]
    cap = params.usable_wh
    if ambient_temp_c is not None:
        cap *= float(battery_capacity_factor(float(ambient_temp_c)))
    return cap
