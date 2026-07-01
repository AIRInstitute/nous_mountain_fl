# Experimentos Flower para balizas de puertos de montaña y autonomía de EV

Este repositorio implementa una prueba de concepto para la idea de **balizas V2I en puertos de montaña sin conectividad externa**. La baliza actúa como servidor federado local; cada coche que cruza el puerto actúa como cliente FL. El objetivo no es predecir punto a punto, sino estimar **al entrar** la energía necesaria para cruzar el tramo y compararla **al salir** con la energía real consumida.

La comparación central es:

```text
Baseline físico plano sin orografía
vs
MLP residual con orografía entrenado centralizado
vs
MLP residual con orografía entrenado con Flower/FedAvg
vs
MLP residual con orografía entrenado con Flower/FedProx
```

El baseline principal es `B1_flat_physics_entry_no_topography`: usa distancia, parámetros físicos nominales, velocidad esperada constante, rodadura, aerodinámica y auxiliares, pero fuerza pendiente cero. Representa el caso “sé cuántos km quedan, pero no conozco la orografía del puerto”.

La red neuronal propuesta no sustituye la física. Aprende un residual:

```text
E_pred_segmento = E_fisica_plana_segmento + MLP(features_topograficas, clima, vehículo)
E_pred_puerto   = suma(E_pred_segmento)
```

## Estructura

```text
configs/
  default.yaml                  # Experimentos base de un split
  quick.yaml                    # Smoke test rápido de run_all
  extended.yaml                 # Validación cruzada + curva de aprendizaje de baliza
  quick_extended.yaml           # Smoke test rápido de run_extended

data/
  raw/                          # Copiar aquí todos los CSV válidos
  expected_files.txt            # Lista orientativa de ficheros esperados

src/mountain_pass_fl/
  data.py                       # Carga, inferencia de metadatos, segmentación
  baselines.py                  # Modelos físicos básicos y nominales
  features.py                   # Normalización y one-hot
  models.py                     # MLP residual
  train.py                      # Entrenamiento centralizado PyTorch
  flower_exp.py                 # FedAvg/FedProx con Flower
  prequential.py                # Evaluación predict-then-update de baliza
  cross_validation.py           # Validación cruzada por rutas/grupos
  learning_curve.py             # Curva de aprendizaje progresivo de la baliza
  risk.py                       # Avisos de riesgo y false-safe rate
  plotting.py                   # Figuras
  run_all.py                    # Orquestador base
  run_extended.py               # Orquestador CV + learning curve

scripts/
  setup_windows.ps1/.bat
  run_all_windows.ps1/.bat
  run_extended_windows.ps1/.bat
  run_extended_windows_no_flower.ps1/.bat
  run_extended_quick_windows.ps1/.bat
```

## Datos esperados

Los CSV deben estar en `data/raw/`. Para la evaluación principal usa solo las rutas completas.

Lista orientativa de rutas completas:

```text
datos_grid_S01_tesla_SOC90_dry24.csv
datos_grid_S02_tesla_SOC90_wet10.csv
datos_grid_S03_tesla_SOC90_snow_neg5.csv
datos_grid_S04_tesla_SOC40_dry24.csv
datos_grid_S05_tesla_SOC40_wet10.csv
datos_grid_S06_tesla_SOC40_snow_neg5.csv
datos_grid_S07_audi_SOC90_dry24.csv
datos_grid_S08_audi_SOC90_wet10.csv
datos_grid_S09_audi_SOC90_snow_neg5.csv
datos_grid_S10_audi_SOC40_dry24.csv
datos_grid_S11_audi_SOC40_wet10.csv
datos_grid_S12_audi_SOC40_snow_neg5.csv
```

Si quieres protegerte contra rutas parciales, edita `configs/extended.yaml`:

```yaml
data_filters:
  min_route_distance_m: 8000
```

Con `0` no aplica filtro.

## Instalación en Windows PowerShell

Desde la carpeta raíz del proyecto, donde están `pyproject.toml`, `requirements.txt` y `src/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

También puedes usar:

```powershell
.\scripts\setup_windows.ps1
```

Si PowerShell bloquea los `.ps1`, usa los `.bat` equivalentes.

## Ejecución base

Ejecuta un único split, baselines, centralizado, Flower y prequential:

```powershell
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_all --data-dir data\raw --out-dir outputs --config configs\default.yaml --rebuild-segments
```

Sin Flower:

```powershell
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_all --data-dir data\raw --out-dir outputs --config configs\default.yaml --rebuild-segments --skip-flower
```

## Ejecución extendida: validación cruzada + curva de baliza

Este es el comando recomendado para los experimentos finales:

```powershell
.\.venv\Scripts\python.exe -m mountain_pass_fl.run_extended --data-dir data\raw --out-dir outputs_extended --config configs\extended.yaml --rebuild-segments
```

O con script:

```powershell
.\scripts\run_extended_windows.ps1
```

Si quieres ejecutar todo excepto Flower, útil para depurar rápido:

```powershell
.\scripts\run_extended_windows_no_flower.ps1
```

Smoke test rápido:

```powershell
.\scripts\run_extended_quick_windows.ps1
```

## Qué añade `run_extended.py`

### 1. Validación cruzada por rutas/clientes

Nunca divide por filas ni por segmentos aleatorios. Cada fold deja fuera clientes/CSV completos.

Protocolos incluidos:

```text
leave_one_route_out
leave_one_scenario_group_out     # vehículo + condición: Tesla_dry, Audi_snow, etc.
leave_one_weather_out            # dry/wet/snow
leave_one_vehicle_out            # Tesla/Audi
repeated_random_client_split     # varios splits train/val/test por cliente
```

Salidas:

```text
outputs_extended/cv/cv_route_predictions.csv
outputs_extended/cv/cv_fold_summary.csv
outputs_extended/cv/cv_summary.csv
outputs_extended/cv/cv_splits.csv
outputs_extended/cv/risk/risk_summary_grouped.csv
```

### 2. Curva de aprendizaje progresivo de la baliza

Simula órdenes de llegada de vehículos:

```text
coche entra -> predice con modelo actual
coche sale -> actualiza la baliza
siguiente coche entra -> usa modelo mejorado
```

Se repite con muchas permutaciones de orden de llegada y con distintos tamaños de lote:

```text
batch_size = 1, 2, 4
```

Así puedes responder:

```text
¿Qué ocurre cuando la baliza ha visto 0, 1, 2, 4, 8 o todos los coches?
```

Salidas:

```text
outputs_extended/learning_curve/learning_curve_route_predictions.csv
outputs_extended/learning_curve/learning_curve_point_summary.csv
outputs_extended/learning_curve/learning_curve_summary.csv
outputs_extended/learning_curve/risk/risk_summary_grouped.csv
```

`remaining_unseen` evalúa sobre coches que todavía no han pasado. Es la métrica más honesta.

`all_routes_replay` reevalúa todas las rutas con el modelo tras ver k coches. Sirve como diagnóstico, no como generalización pura cuando k > 0.

### 3. Checkpoints Flower explícitos

`learning_curve.py` incluye una simulación secuencial rápida de FedAvg/FedProx. Además, `run_extended.py` puede lanzar checkpoints con Flower real:

```text
primeros 2 coches vistos
primeros 4 coches vistos
primeros 8 coches vistos
todos los coches vistos
```

Salidas:

```text
outputs_extended/learning_curve/flower_checkpoints/flower_checkpoints_route_predictions.csv
outputs_extended/learning_curve/flower_checkpoints/flower_checkpoints_summary.csv
```

Estos checkpoints son más lentos porque cada punto arranca una simulación Flower/Ray. Se pueden desactivar con `--skip-flower`.

## Salidas principales

```text
outputs_extended/segments.csv
outputs_extended/client_metadata.csv
outputs_extended/cv/cv_summary.csv
outputs_extended/learning_curve/learning_curve_summary.csv
outputs_extended/learning_curve/flower_checkpoints/flower_checkpoints_summary.csv
outputs_extended/plots/
outputs_extended/EXTENDED_REPORT.md
```

Figuras relevantes:

```text
outputs_extended/plots/cv_mae_boxplot_*.png
outputs_extended/plots/learning_curve_mae_remaining_unseen.png
outputs_extended/plots/learning_curve_false_safe_remaining_unseen.png
```

## Métricas importantes

```text
MAE energía total [kWh]
RMSE energía total [kWh]
Bias [Wh]
MAPE [%]
MAE SOC final [%]
false-safe rate
false-warning rate
risk recall
```

La métrica más alineada con la motivación del artículo es:

```text
false-safe rate = el sistema dice “puedes cruzar”, pero realmente termina por debajo de la reserva
```

## Nota metodológica

El modelo FL no recibe `power_watts`, `current_a`, `voltage_v`, `energy_used_cum_wh` ni `energy_regen_cum_wh` como entrada. Esas columnas solo se usan para construir la etiqueta real de energía. Así se evita fuga de información.

La red neuronal recibe features topográficas y climáticas y aprende el residual respecto al baseline físico plano. Por tanto, el argumento experimental no es “la NN sustituye a la física”, sino:

```text
la baliza aprende una corrección local y específica del puerto sobre un estimador físico/general.
```
