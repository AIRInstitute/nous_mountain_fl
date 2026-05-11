# Simulación energética de VE en el Puerto de la Quesera — Dataset experimental

Este repositorio contiene un dataset experimental factorial producido por la simulación de un gemelo digital de un vehículo eléctrico (VE) atravesando un fragmento representativo del **Puerto de la Quesera** (Sistema Central, carretera GU-186). El dataset comprende 12 escenarios resultantes de la combinación factorial de vehículo, estado inicial de carga (SOC) y condición climática.

El pipeline es **DEM → MATLAB → RoadRunner → CARLA Sim 0.9.16 → modelo energético en Python**, con trazas cinemáticas del vehículo alimentadas a un modelo energético longitudinal con acoplamiento climático (HVAC, resistencia a la rodadura, capacidad de batería, frenado regenerativo).

Estos datos sustentan un artículo en preparación para *Open Research Europe (ORE)*, donde se reporta la metodología del gemelo digital y los resultados del caso de estudio.

---

## 1. Contexto geográfico y topográfico

| Propiedad | Valor |
|---|---|
| Región | Sistema Central (España), frontera entre Guadalajara y Segovia |
| Carretera | GU-186 (OSM way `157020458`, clasificada como *tertiary*) |
| Nombre de referencia | Puerto de la Quesera |
| Sistema de coordenadas | WGS84 / EPSG:4326 |
| Bounding box (DEM) | lon `[-3.4001°, -3.3540°]`, lat `[41.1610°, 41.1821°]` |
| Dimensiones del DEM | 3.86 km × 2.35 km (9.08 km²) |
| Fuente del DEM | Copernicus COP30 (~30 m de resolución horizontal) |
| Rango de elevación del terreno | 1203 m a 1798 m s.n.m. (595 m de desnivel) |
| Longitud de carriles navegables en HD map | 15.04 km (truncados a una única ruta sin bifurcaciones entre dos rotondas para garantizar runs deterministas) |
| Trayectoria del vehículo por run | ~10 km de carretera serpenteante de montaña |

**Nota sobre la elevación del mapa:** la red OpenDRIVE exportada desde RoadRunner tiene elevación plana en los carriles (`z=0`), mientras que el mesh visual y los collisionadores físicos conservan la elevación del DEM. En consecuencia, el modelo VE lee la pendiente desde el **ángulo de pitch instantáneo del vehículo**, no desde la red de waypoints. Este enfoque se validó contra predicciones de literatura para consumo energético inducido por pendiente.

---

## 2. Diseño experimental

Un **diseño factorial 2 × 2 × 3** con 12 escenarios:

- **Factor 1:** Vehículo (Tesla Model 3 RWD, Audi e-tron 55 quattro)
- **Factor 2:** Estado inicial de carga (90%, 40%)
- **Factor 3:** Clima (dry/24°C, wet/10°C, snow/-5°C)

Cada run consiste en una traversía de ~10 km del segmento truncado de la GU-186, con el Traffic Manager de CARLA conduciendo el vehículo autónomamente y registrando datos cinemáticos a 20 Hz. Las magnitudes energéticas se computan en Python desde la traza cinemática usando un modelo VE personalizado.

### 2.1 Parámetros de los vehículos

Los parámetros están almacenados en `ev_model.py` bajo el diccionario `EV_LIBRARY`. Fuentes: especificaciones del fabricante, datos de pruebas EPA, ev-database.org, pruebas ADAC.

| Parámetro | Tesla Model 3 RWD | Audi e-tron 55 quattro |
|---|---|---|
| Masa en vacío (kg) | 1611 | 2490 |
| Coeficiente de arrastre $C_d$ | 0.23 | 0.27 |
| Área frontal $A$ (m²) | 2.22 | 2.65 |
| Coef. resistencia rodadura $C_{rr}$ | 0.011 | 0.012 |
| Voltaje nominal del pack (V) | 350 | 396 |
| Capacidad útil de batería (kWh) | 72.0 | 86.5 |
| Eficiencia motor $\eta_{motor}$ | 0.90 | 0.88 |
| Eficiencia regen $\eta_{regen}$ | 0.85 | 0.82 |
| Potencia máxima regen (kW) | 60 | 220 |
| Potencia máxima descarga (kW) | 210 | 300 |
| Resistencia interna pack (Ω) | 0.07 | 0.06 |
| Masa térmica batería (J/K) | 180000 | 230000 |
| Disipación batería (W/K) | 250 | 320 |
| Potencia HVAC base (W) | 300 | 400 |

### 2.2 Parámetros climáticos

Tres condiciones climáticas, cada una parametrizada consistentemente entre los WeatherParameters de CARLA y las funciones de acoplamiento climático del modelo VE:

| Tag de clima | $T_{amb}$ (°C) | Condición pavimento | Precipitación | Wetness | Viento | Altitud del sol |
|---|---|---|---|---|---|---|
| `dry24` | 24 | dry | 0% | 0% | 0% | 30° |
| `wet10` | 10 | wet | 60% | 70% | 30% | 10° |
| `snow_neg5` | -5 | snow | 80% | 40% | 50% | 5° |

**Efectos del clima en el modelo VE:**

El modelo energético aplica tres multiplicadores según el clima:

1. **Potencia auxiliar HVAC** $P_{aux}(T_{amb})$: lineal a tramos, 500 W a 24°C → 1500 W a 10°C → 3750 W a -5°C
2. **Factor de capacidad de batería** (Liu et al., 2018 para Li-ion NMC): 1.00 a 24°C → 0.95 a 10°C → 0.86 a -5°C
3. **Resistencia a la rodadura efectiva** (Sandberg, 2011): ×1.0 dry, ×1.2 wet, ×2.0 snow

**Nota sobre renderizado visual:** CARLA 0.9.16 no renderiza partículas de nieve de forma nativa. Los escenarios "snow" se caracterizan por tanto mediante sus *efectos físicos* sobre el modelo VE (baja $T_{amb}$, alto $C_{rr}$, capacidad de batería reducida, HVAC alto), no visualmente. El renderizado de lluvia y la humedad superficial sí están presentes visualmente en los escenarios `wet10`.

### 2.3 Catálogo completo de escenarios

| ID | Vehículo | SOC | Clima | $T_{amb}$ | Pavimento | $P_{precip}$ | Wetness | Wind | Sol |
|---|---|---|---|---|---|---|---|---|---|
| S01 | Tesla Model 3 | 90% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S02 | Tesla Model 3 | 90% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S03 | Tesla Model 3 | 90% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |
| S04 | Tesla Model 3 | 40% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S05 | Tesla Model 3 | 40% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S06 | Tesla Model 3 | 40% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |
| S07 | Audi e-tron | 90% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S08 | Audi e-tron | 90% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S09 | Audi e-tron | 90% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |
| S10 | Audi e-tron | 40% | dry24 | 24°C | dry | 0% | 0% | 0% | 30° |
| S11 | Audi e-tron | 40% | wet10 | 10°C | wet | 60% | 70% | 30% | 10° |
| S12 | Audi e-tron | 40% | snow_neg5 | -5°C | snow | 80% | 40% | 50% | 5° |

---

## 3. Descripción del dataset

Cada escenario produce un archivo CSV llamado `datos_grid_<ID>.csv` con una fila por tick de simulación (20 Hz). Los 12 archivos comparten las mismas 41 columnas.

### 3.1 Convención de nombrado

```
datos_grid_S01_tesla_SOC90_dry24.csv
datos_grid_S02_tesla_SOC90_wet10.csv
...
datos_grid_S12_audi_SOC40_snow_neg5.csv
```

Cada archivo es aproximadamente 25,000 filas (la trayectoria simulada dura ~21 minutos a 20 Hz).

### 3.2 Esquema de columnas

| Columna | Unidades | Descripción |
|---|---|---|
| `scenario_id` | string | Identificador completo del escenario (ej. `S01_tesla_SOC90_dry24`) |
| `vehicle_key` | string | Clave en `EV_LIBRARY` (`tesla_model3`, `audi_etron`) |
| `run_id` | string | UUID de 8 caracteres único por run, útil si un escenario se re-ejecuta |
| `vehicle_id` | int | ID del actor en CARLA |
| `timestamp_abs` | s | Tiempo absoluto del simulador (desde el inicio del simulador) |
| `time_relative` | s | Tiempo desde el inicio de este run |
| `dt` | s | Duración del tick (constante 0.05 s) |
| `road_id`, `lane_id` | int | Identificadores OpenDRIVE (informativo; el mapa es z=0) |
| `pos_x`, `pos_y`, `pos_z` | m | Posición del vehículo en el frame mundial de CARLA. `pos_z` refleja la elevación real desde el mesh del DEM |
| `heading` | grados | Ángulo de yaw del vehículo |
| `road_grade_deg` | grados | Pendiente de la carretera, leída del pitch del vehículo (positivo = subiendo). Rango observado: −19° a +20° |
| `road_curvature` | 1/m | Curvatura instantánea, computada de la velocidad angular |
| `road_friction` | — | Parámetro de fricción de neumáticos en CARLA |
| `speed_kmh` | km/h | Velocidad del vehículo |
| `acc_long`, `acc_lat` | m/s² | Aceleración proyectada en los vectores forward y right del vehículo |
| `throttle`, `brake`, `steering` | 0..1 / -1..1 | Entradas de control del autopilot |
| `gear` | int | Marcha engranada |
| `soc_percent` | % | SOC instantáneo |
| `capacity_usable_wh` | Wh | Capacidad utilizable efectiva (depende de $T_{amb}$) |
| `voltage_v` | V | Voltaje instantáneo del pack |
| `current_a` | A | Corriente instantánea del pack (positiva = descarga, negativa = carga) |
| `power_watts` | W | Potencia neta en la batería (positiva = descarga, negativa = regen) |
| `energy_used_cum_wh` | Wh | Energía cumulativa consumida desde el inicio del run |
| `energy_regen_cum_wh` | Wh | Energía cumulativa regenerada desde el inicio del run |
| `is_regen` | 0/1 | 1 si la potencia es negativa este tick (regenerando) |
| `battery_temp_c` | °C | Temperatura modelada de la batería |
| `aux_power_w` | W | Potencia HVAC/auxiliar instantánea aplicada |
| `c_rr_eff` | — | Coeficiente de resistencia a la rodadura efectivo |
| `ambient_temp_c` | °C | Temperatura ambiente del escenario (constante por run) |
| `precip_type` | string | `none` o `rain` |
| `precip_intensity` | 0..100 | Parámetro de precipitación del escenario |
| `wind_speed_ms` | m/s | Velocidad del viento del escenario |
| `wind_dir_deg` | grados | Dirección del viento del escenario |
| `wind_long_ms` | m/s | Componente longitudinal del viento (positivo = viento en contra) |
| `road_condition` | string | `dry`, `wet`, o `snow` |

### 3.3 Terminación del run

Un run termina cuando se cumple una de las siguientes condiciones:

1. **Objetivo de distancia alcanzado:** el vehículo ha acumulado ≥ 10,000 m de viaje físico (la ruta truncada es ~10.5 km de un extremo a otro).
2. **Detector de atasco:** el vehículo se ha movido menos de 10 m en una ventana de 30 s (es decir, está atascado en una rotonda u obstáculo). En este caso `stop_reason = stuck`.
3. **Timeout de seguridad:** 3600 s de tiempo simulado (60 min). Nunca activado en este dataset.

Los 12 runs en este dataset terminaron normalmente por la condición 1 (`reached_destination`).

---

## 4. Tabla maestra de resultados

Todos los runs alcanzaron el objetivo de 10 km en ~1250 s de tiempo simulado, viajando entre elevaciones de 1242 m (valle) y ~1460 m (zona de rotonda superior), con un ascenso acumulado de ~412 m y descenso de ~254 m. La ruta es serpenteante, lo que proporciona una amplia distribución de bins de pendiente por run (de −19° a +20°).

| # | Vehículo | SOC ini | Clima | SOC fin | Caída SOC | E usada | E regen | E neta | Wh/km |
|---|---|---|---|---|---|---|---|---|---|
| S01 | Tesla | 90% | dry 24°C | 87.65% | 2.35 pts | 2472 Wh | 781 Wh | 1691 Wh | 169.1 |
| S02 | Tesla | 90% | wet 10°C | 86.93% | 3.07 pts | 2761 Wh | 663 Wh | 2099 Wh | 209.9 |
| S03 | Tesla | 90% | snow -5°C | 84.58% | 5.42 pts | 3781 Wh | 418 Wh | 3363 Wh | 336.3 |
| S04 | Tesla | 40% | dry 24°C | 37.65% | 2.35 pts | 2472 Wh | 781 Wh | 1691 Wh | 169.1 |
| S05 | Tesla | 40% | wet 10°C | 36.76% | 3.24 pts | 2880 Wh | 663 Wh | 2217 Wh | 221.7 |
| S06 | Tesla | 40% | snow -5°C | 34.58% | 5.42 pts | 3781 Wh | 418 Wh | 3363 Wh | 336.3 |
| S07 | Audi | 90% | dry 24°C | 86.73% | 3.27 pts | 3897 Wh | 1068 Wh | 2829 Wh | 282.9 |
| S08 | Audi | 90% | wet 10°C | 86.00% | 4.00 pts | 4210 Wh | 924 Wh | 3286 Wh | 328.6 |
| S09 | Audi | 90% | snow -5°C | 83.77% | 6.23 pts | 5269 Wh | 623 Wh | 4646 Wh | 464.6 |
| S10 | Audi | 40% | dry 24°C | 36.73% | 3.27 pts | 3897 Wh | 1068 Wh | 2829 Wh | 282.9 |
| S11 | Audi | 40% | wet 10°C | 36.10% | 3.90 pts | 4122 Wh | 915 Wh | 3207 Wh | 320.7 |
| S12 | Audi | 40% | snow -5°C | 33.72% | 6.28 pts | 5296 Wh | 612 Wh | 4684 Wh | 468.4 |

### 4.1 Matriz de energía neta (Wh por 10 km de traversía)

|  | dry 24°C | wet 10°C | snow -5°C |
|---|---|---|---|
| **Tesla SOC 90%** | 1691 | 2099 | 3363 |
| **Tesla SOC 40%** | 1691 | 2217 | 3363 |
| **Audi SOC 90%**  | 2829 | 3286 | 4646 |
| **Audi SOC 40%**  | 2829 | 3207 | 4684 |

### 4.2 Matriz de caída de SOC (puntos porcentuales por 10 km)

|  | dry 24°C | wet 10°C | snow -5°C |
|---|---|---|---|
| **Tesla SOC 90%** | 2.35 | 3.07 | 5.42 |
| **Tesla SOC 40%** | 2.35 | 3.24 | 5.42 |
| **Audi SOC 90%**  | 3.27 | 4.00 | 6.23 |
| **Audi SOC 40%**  | 3.27 | 3.90 | 6.28 |

---

## 5. Hallazgos principales

### 5.1 El clima tiene un efecto mayor que la elección del vehículo

La energía neta en nieve puede ser aproximadamente **el doble** que en condiciones secas:

- Tesla: snow consume **+99% más** que dry (1691 → 3363 Wh)
- Audi: snow consume **+64% más** que dry (2829 → 4646 Wh)

El Tesla es *más* sensible al clima en términos relativos porque su línea base es menor, por lo que las mismas penalizaciones de HVAC y resistencia a la rodadura pesan más sobre su consumo proporcionalmente más pequeño. Esto sugiere que **los VE eficientes experimentan penalizaciones inducidas por clima proporcionalmente mayores**, una idea que puede ser de interés al extrapolar la planificación de autonomía a nivel de flota para corredores de montaña bajo condiciones adversas.

### 5.2 La dominancia del vehículo se atenúa con la severidad del clima

El Audi consume mucho más que el Tesla en condiciones secas, pero la brecha se reduce en nieve:

| Clima | Ratio Audi / Tesla | Brecha |
|---|---|---|
| dry | 1.67× | +67.3% |
| wet | 1.57× | +56.6% |
| snow | 1.38× | +38.1% |

Esto ocurre porque la carga inducida por el clima (HVAC, $C_{rr}$, reducción de capacidad) es en gran medida independiente del vehículo en términos absolutos, por lo que aparece como una penalización relativa más pequeña sobre el ya mayor consumo del Audi. La potencia regen pico del e-tron (220 kW vs 60 kW para el Model 3) también contribuye a cerrar la brecha en escenarios con descensos significativos.

### 5.3 El SOC inicial apenas afecta el consumo neto

Un hallazgo central para la validación física del modelo: bajo condiciones equivalentes (mismo vehículo, mismo clima), el SOC al que comienza la trayectoria **no** altera la energía neta consumida por la trayectoria. Esto se observa en:

- S04 ≡ S01 (Tesla dry): ambos 1691 Wh
- S06 ≡ S03 (Tesla snow): ambos 3363 Wh
- S10 ≡ S07 (Audi dry): ambos 2829 Wh
- S12 ≈ S09 (Audi snow): 4684 vs 4646 Wh (Δ = +38 Wh, +0.8%)

Solo en condiciones `wet` emerge un pequeño efecto, con **signos opuestos** entre vehículos:

- S05 vs S02 (Tesla wet): +119 Wh (+5.65%) al arrancar a SOC 40%
- S11 vs S08 (Audi wet, media de dos re-runs): −105 Wh (−3.2%) al arrancar a SOC 40%

Esta asimetría entre vehículos se explica por la interacción de:
- Pérdidas $I^2 R$ (favorecen SOC 90% porque mayor voltaje del pack significa menor corriente para la misma potencia)
- Límites de saturación del frenado regenerativo (favorecen SOC 40% en vehículos con alta capacidad regen, porque menor voltaje permite absorber más potencia mecánica antes de saturar `max_regen_power`)

Para el Tesla (techo de 60 kW regen), el efecto $I^2 R$ domina → consumo neto *mayor* a SOC 40%.
Para el Audi (techo de 220 kW regen), el efecto de alivio de saturación domina → consumo neto *menor* a SOC 40%.

Esta es una observación empírica no trivial que motiva modelado más rico, consciente del powertrain, en trabajo futuro.

### 5.4 Cuantificación de range-anxiety (escenarios SOC 40%)

Proyectando la tasa de caída de SOC observada en 10 km de carretera de montaña a trayectorias más largas, asumiendo que las condiciones climáticas y la distribución de pendientes se mantienen similares:

| Escenario | Caída SOC en 10 km | km proyectados hasta SOC = 10% (condiciones de montaña) |
|---|---|---|
| Tesla SOC 40% dry | 2.35 pts | 128 km |
| Tesla SOC 40% wet | 3.24 pts | 83 km |
| Tesla SOC 40% snow | 5.42 pts | **55 km** |
| Audi SOC 40% dry | 3.27 pts | 82 km |
| Audi SOC 40% wet | 3.90 pts | 67 km |
| Audi SOC 40% snow | 6.28 pts | **48 km** |

Los dos peores escenarios (S06 y S12) implican que un conductor que comenzó un puerto de montaña de 10 km a SOC 40% en nieve podría continuar **38–45 km adicionales** antes de alcanzar la reserva de seguridad convencional del 10%. Esto es un rango mucho menor que el que sugeriría una estimación ingenua basada en consumo de autopista, motivando la necesidad de sistemas de aviso de energía conscientes del terreno y del clima (como el sistema basado en V2X que el proyecto más amplio aspira a desarrollar).

---

## 6. Reproducibilidad

El Traffic Manager de CARLA introduce una pequeña cantidad de estocasticidad en el control de throttle y dirección entre runs. Para evaluarlo:

- **S11 se ejecutó dos veces** (run 1 y run 2) bajo parámetros de escenario idénticos.
- Energía neta: 3207 Wh (run 1) vs 3156 Wh (run 2)
- Variabilidad: ±25 Wh sobre una media de 3181 Wh, **CV ≈ 0.8%**

Esto confirma que las diferencias escenario-a-escenario reportadas en este dataset (típicamente decenas a cientos de Wh) están dominadas por los factores experimentales controlados, no por ruido del autopilot.

---

## 7. Contenidos del repositorio

```
.
├── README.md                              # este documento
├── ev_model.py                            # modelo energético VE con biblioteca de vehículos
├── seccionA_B_C_D_E_v5_2.py               # script de adquisición CARLA con catálogo de 12 escenarios
├── grid_runs/
│   ├── datos_grid_S01_tesla_SOC90_dry24.csv
│   ├── datos_grid_S02_tesla_SOC90_wet10.csv
│   ├── datos_grid_S03_tesla_SOC90_snow_neg5.csv
│   ├── datos_grid_S04_tesla_SOC40_dry24.csv
│   ├── datos_grid_S05_tesla_SOC40_wet10.csv
│   ├── datos_grid_S06_tesla_SOC40_snow_neg5.csv
│   ├── datos_grid_S07_audi_SOC90_dry24.csv
│   ├── datos_grid_S08_audi_SOC90_wet10.csv
│   ├── datos_grid_S09_audi_SOC90_snow_neg5.csv
│   ├── datos_grid_S10_audi_SOC40_dry24.csv
│   ├── datos_grid_S11_audi_SOC40_wet10.csv
│   └── datos_grid_S12_audi_SOC40_snow_neg5.csv
└── 
```

---

## 8. Cómo leer un CSV en Python

```python
import pandas as pd

df = pd.read_csv('grid_runs/datos_grid_S03_tesla_SOC90_snow_neg5.csv')

print(f"Escenario: {df['scenario_id'].iloc[0]}")
print(f"Vehículo:  {df['vehicle_key'].iloc[0]}")
print(f"Duración:  {df['time_relative'].iloc[-1]:.1f} s")
print(f"SOC: {df['soc_percent'].iloc[0]:.2f}% -> {df['soc_percent'].iloc[-1]:.2f}%")
print(f"Energía neta: {df['energy_used_cum_wh'].iloc[-1] - df['energy_regen_cum_wh'].iloc[-1]:.0f} Wh")

# Consumo energético estratificado por pendiente de carretera
import numpy as np
bins = [-20, -10, -5, -1, 1, 5, 10, 20]
df['grade_bin'] = pd.cut(df['road_grade_deg'], bins=bins)
df['e_step'] = df['power_watts'] * df['dt'] / 3600.0   # Wh por tick
# (calcular distancia recorrida por tick por separado para obtener Wh/km por bin)
```

---

## 9. Agradecimientos y referencias

**Datos geoespaciales:** DEM de Copernicus COP30 (Agencia Espacial Europea); geometría de carreteras de OpenStreetMap.

**Fuentes de calibración del modelo:**
- Liu, K., et al. (2018). *A brief review on key technologies in the battery management system of electric vehicles*. Frontiers of Mechanical Engineering. (Capacidad de batería vs temperatura)
- Sandberg, U. (2011). *Rolling resistance — basic information and state-of-the-art on measurement methods*. (Multiplicadores de resistencia a la rodadura bajo condiciones wet/snow)

**Software:** CARLA Sim 0.9.16 (Unreal Engine 4), MathWorks RoadRunner R2024a, MATLAB R2024a, Python 3.12 con NumPy, Pandas.

---

## 10. Contacto

Repositorio mantenido por Emmanuel Cuevas, con colaboradores Yeray Mezquita Martín, Albano Carrera Gonzáles y Diego Valdeolmillos Villaverde. Para preguntas sobre datos, supuestos del modelo o diseño experimental, por favor abrir un issue en este repositorio.
