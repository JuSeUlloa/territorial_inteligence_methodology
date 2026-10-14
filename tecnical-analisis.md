# Análisis Técnico del Flujo — Territorial Intelligence (Hurtos de Personas)

> Documento técnico del pipeline de inteligencia territorial: tratamiento de datos,
> variables del modelo, justificación de selección y hallazgos priorizados.

---

## 1. Arquitectura end-to-end

```
Parquet (Volumen) ─► BRONZE ─► SILVER ─► MODEL ─► GOLD ─► DASHBOARD
                     (crudo +   (malla      (XGBoost  (tasas,   (AI/BI
                      encode)    mensual)     + pred)   umbrales)  3 hojas)
```

| Paso | Código | Tabla de salida | Granularidad |
|---|---|---|---|
| `quality` | `bronze/quality.py` | — (reporte) | — |
| `encode` | `bronze/bronce.py` | `hurto_personas_bronce` (735,802) | 1 fila = hecho (categoría × fecha) |
| `silver` | `silver/silver.py` → `utils/domains.py` | `hurto_personas_silver` | 1 fila = **municipio × mes** (malla completa) |
| `model` | `models/xgboots_model.py` | `hurto_personas_prediccion` / `hurto_personas_forecast` | 1 fila = municipio × mes **predicho** |
| `consolidated` | `gold/consolidated.py` | `thresholds`, `hurtos_personas_consolidado`, `hurtos_personas_gold` | tasas, errores, umbrales, geometría |

Entorno: Databricks (`workspace.territorial_inteligence`), CLI v1.14.1, Python 3.12, uv.

Tablas de apoyo: `holidays` (2021-2026, 18 festivos/año), `municipios_geo` (1,152),
`departamentos_geo` (33), `consolidado` (1,120 municipios con variables estructurales).

---

## 2. Tratamiento de datos por capa

### 2.1 BRONZE — carga y codificación

**Carga** (`bronze/loader.py:73`): Parquet → Volume → `CREATE OR REPLACE TABLE ... USING DELTA AS SELECT`,
con renombrado de columnas vía `COLUMNS_MAPPING` (`Temática→tematica`, `Código Dane→codigo_dane`, etc.).

**Calidad** (`bronze/quality.py`): conteo de nulos, valores distintos, duplicados exactos y
estadísticos de `anio`/`cantidad`. Es **informativo**, no bloqueante: no aborta el pipeline si
encuentra nulos.

**Codificación** (`bronze/bronce.py:23-56`) — punto crítico del flujo:

- `genero`, `zona`, `armas_medios`, `agrupa_edad`: `SELECT DISTINCT` ordenado alfabéticamente →
  código `idx + 1` en un `CASE` regenerado **en cada ejecución**.
- `mes`: mapping fijo en `domains.py` (Enero→1 … Diciembre→12).

> **Riesgo**: los códigos son dependientes de los datos. Silver interpreta `zona=3` (urbana),
> `genero=2` (masculino), `armas_medios=3` (fuego) con `CASE` **hardcodeados**
> (`domains.py:62-72`). Si aparece un valor nuevo en el origen, el orden alfabético se altera,
> los códigos se desplazan silenciosamente y los pivotes de composición quedan incorrectos sin
> que se produzca ningún error visible.
> **Recomendación**: congelar el mapping en `domains.py` o persistirlo en una tabla de dimensión.

### 2.2 SILVER — el corazón del tratamiento (`utils/domains.py`, 13 CTEs)

```
agregacion ──► municipios ─┐
   │                       ├─► spine ──► base_completa ──► composicion ──► temporal ──► features ─┐
   └──► rango_fechas ──► calendario ──► periodos ──┘      (LEFT JOIN)    (pct_*)     (ventanas)   ├──► SELECT final
                                                                                                  │   + festividades
```

| CTE | Qué hace | Detalle del tratamiento |
|---|---|---|
| `agregacion` | Normaliza `codigo_dane` + agrega | CASE por longitud (8→LEFT 5; 7→quita sufijo de 3 dígitos + LPAD 5; 4→LPAD). `SUM(cantidad)=total_tematica` + 24 pivotes SUM condicional. **Grupo: codigo_dane × mes × anio** |
| `calendario` / `periodos` | Secuencia mensual `SEQUENCE(MIN, MAX, 1 MONTH)` | genera el calendario completo del rango de datos |
| `spine` | `municipios CROSS JOIN periodos` | **malla completa**: todo municipio tiene fila cada mes aunque no haya hurtos |
| `base_completa` | `spine LEFT JOIN agregacion` + flag `hay_observacion` | **1 = dato observado; 0 = sin fila → los totales quedan NULL, NO 0** (NULL ≠ cero medido) |
| `composicion` | `pct_* = categoría / total * 100` | solo si `total > 0`; si no → NULL (evita división por cero) |
| `temporal` | Ventanas `PARTITION BY codigo_dane ORDER BY anio, mes` | lags 1/2/3/6/12; `n_obs_3/6/12` = COUNT de no-nulos en ventana; `historia_meses` = COUNT total del municipio; `media_*_raw`; `max_12_meses`; `pct_*_lag_1` |
| `features` | Gating de medias móviles | `media_movil_3/6/12` **solo se emite si `n_obs = N` exacto** → ventana sin huecos; si no → NULL + flag `historia_insuficiente_*` |
| `festividades` | `holidays` agregado a `n_festivos` / `tiene_festivo` | solo para dashboard, **no entra al modelo** |

**Tratamientos de integridad implementados:**

- Cobertura explícita (`n_obs_*`) en vez de imputar — XGBoost consume NULL nativamente.
- Medias sin histórico suficiente → NULL + indicador, no promedio engañoso.
- `hay_observacion` separa "cero real" de "mes sin dato"; el entrenamiento filtra `hay_observacion = 1`.
- `codigo_dane` normalizado **una sola vez** en `agregacion`; aguas abajo solo se hace `CAST`.

**Normalización de `codigo_dane`**: códigos fuente de 7-8 dígitos (ej. `5001000`, `5001012` =
corregimientos de MEDELLÍN) → se recortan a 5 dígitos (LEFT de longitud−3) y se rellenan con
LPAD. Resultado: `05001`.

> **Bugs activos conocidos**: el `SELECT` final pide `tendencia`, `crecimiento_intermensual`,
> `crecimiento_interanual`, `max_anio_anterior` y `max_mes_anterior` que **no se crean** en
> ninguna CTE (causa el error `UNRESOLVED_COLUMN`); la CTE `indexada` queda huérfana porque
> `temporal` lee de `composicion`.

### 2.3 ORGANIZACIÓN ACTUAL DEL CÓDIGO

| Archivo | Rol |
|---|---|
| `main.py` | Orquestador CLI: `--step quality/encode/silver/model/consolidated/all` |
| `bronze/loader.py` | Carga Parquet → Delta, ensure catalog/schema/volume |
| `bronze/bronce.py` | Encoding de categóricos a códigos numéricos |
| `bronze/quality.py` | Reporte de calidad |
| `bronze/schema.py` | Renombrado de columnas (`COLUMNS_MAPPING`) |
| `silver/silver.py` | Ejecuta `STRUCTURE_SILVER_TABLE` como f-string (65 líneas) |
| `utils/domains.py` | SQL completo de Silver + mappings (419 líneas) |
| `models/xgboots_model.py` | Entrenamiento, predicción, forecast, escritura de tablas (506 líneas) |
| `gold/consolidated.py` | Umbrales P25/P75, tabla de error, tabla gold |

---

## 3. Variables del modelo y POR QUÉ se eligieron

**Target**: `tasa_real = (total_tematica / pob_dane_2025) × 100000`
(`xgboots_model.py:131`)

— tasa por 100k habitantes, **no conteo**: para que Bogotá no domine por escala y las tasas sean
comparables entre municipios de tamaño distinto.

**Conjunto**: 48 features = 31 de Silver + 16 de consolidado + 1 derivada.

### 3.1 Bloque temporal autorregresivo (Silver) — 10 vars

| Variable | Razón de selección |
|---|---|
| `total_lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` | **Es la señal más fuerte** (importancia: `total_lag_1` = 18.7%). Los hurtos tienen persistencia mes a mes; lo observado en t−1 es el mejor predictor de t. **Disponible al momento de predecir** (es t−1) → sin fuga de información. |
| `media_movil_3`, `_6`, `_12` | Suavizan el ruido mensual y capturan la línea base estacional (trimestral / semestral / mismo mes del año anterior). Importancia combinada ≈ 25%. Se incluyen **gated** por `n_obs` porque la malla tiene huecos: un promedio sobre 2 de 12 meses reales sería una mentira estadística. |
| `pct_zona_urbana/rural_lag_1` | **Perfil de composición al corte t−1**: distribución urbano/rural. Cambios en el patrón territorial preceden cambios en volumen. Van **rezagadas** → disponibles en pronóstico; la versión actual (sin lag) se excluyó por fuga (usa el total del mes que se predice en su numerador). |
| `pct_arma_fuego/blanca/contundente_lag_1` | **Modus operandi**: una migración hacia arma de fuego anticipa alzas en la tasa. Misma lógica de rezago por disponibilidad. |
| `pct_genero_masculino/femenino_lag_1` | Perfil de víctima; cambia con la moda del crimen (ej. robos a mujeres con celulares en zonas específicas). |
| `pct_edad_adultos/jovenes/menores_lag_1` | Segmento etario víctima; distingue hurtos escolares vs. laborales vs. callejeros. |
| `mes`, `anio` | Estacionalidad de calendario: diciembre/vacaciones suben hurtos, febrero baja. Los árboles capturan `mes` con splits; no requiere one-hot. |

### 3.2 Bloque de cobertura del histórico — 7 vars

`n_obs_3`, `n_obs_6`, `n_obs_12`, `historia_meses`,
`historia_insuficiente_3`, `historia_insuficiente_6`, `historia_insuficiente_12`

**Por qué**: 930 municipios tienen < 60 meses de historia. Sin estas variables, el modelo
trataría `media_movil_12 = 40` igual venga de 12 meses reales o de una historia parcial.
Le dicen al modelo **cuánta evidencia respalda cada feature** → aprende a desconfiar de ventanas
incompletas. También resuelven el problema de `LAG(12)`, que no es "mismo mes del año anterior"
cuando faltan meses intermedios.

### 3.3 Bloque momentum / desviación — 4 vars

`crecimiento_intermensual`, `crecimiento_interanual`, `max_mes_anterior`, `max_anio_anterior`

**Por qué**: capturan **aceleración** (si el municipio viene subiendo o bajando) normalizando por
escala: un ±10% es igual de informativo en un pueblo de 8k que en una ciudad de 100k. Las dos
últimas miden desviación absoluta frente al mes / año de referencia.

> **Problema conocido**: actualmente se calculan con `total_tematica` **del mes que se predice**
> → fuga en entrenamiento y versión shifteada en forecast (train/serve skew). Plan de fix:
> redefinir con rezagos (`lag_1 vs lag_2`, `lag_1 vs lag_12`).

### 3.4 Bloque estructural estático (consolidado) — 16 vars

| Variable | Por qué se incluyó |
|---|---|
| `pob_dane_2025` | **Es el denominador del target** — condiciona directamente la tasa. Importancia 6.8%. |
| `log_pob_dane` | La población es fuertemente asimétrica (8k → 11M); `LOG1P` comprime la cola para que el árbol no haga splits degenerados. Mejoró métricas: RMSE 14.27→14.14, MAE 8.32→8.11. Importancia 4.5%. |
| `codigo_departamento` | Efecto fijo regional: policía, tasas de denuncia y cultura de reporte varían por departamento. |
| `pdet` | Municipios PDET (afectados por conflicto armado): más hurtos y/o menos denuncia. Codifica contexto estructural ausente en la serie temporal. |
| `zomac`, `pdet_zomac` | Zonas de rezago socioeconómico y cruce PDET×ZOMAC: segmentación de política pública. |
| `riesgo_victimizacion` | Índice continuo 0-1 de riesgo de victimización por departamento/municipio. |
| `pres_grup_armados` | Presencia de grupos armados: altera tanto el crimen como la **propensión a denunciar** (sesgo de reporte). 1,001 de 1,120 con dato. |
| `pib2024_...valor_agregado_departamental` | Contexto económico: la delincuencia oportunista correlaciona con la actividad económica. |
| `art95_num1_...ley_1453_2011` | Base de datos negativa rural — proxy de ruralidad/cobertura. **Importancia 14.5% (3ª más alta) → señal de posible proxy de población; pendiente de auditoría de temporalidad.** |
| `longitud`, `latitud` | **Proxy espacial**: sin features geográficas complejas, el árbol aprende regionalización (costa vs. interior, Eje Cafetero) por coordenadas. |
| `cluster_rv` | Categórica ordinal (Bajo→Alto) que **comprime** el índice de riesgo de victimización: robusta a outliers y usa `enable_categorical=True` de XGBoost. 100% cobertura. |
| `cluster_pga`, `cluster_vs` | Ídem para presencia de grupos armados y victimización por seguridad. **119 NULL** (se debe filtrar o avisar en dashboard). |
| `municipio_isla_area_no_municipalizada` | Islas y áreas no municipalizadas: dinámica distinta (turismo, aislamiento). Binaria. |

### 3.5 Variables excluidas y por qué

| Variable | Razón de exclusión |
|---|---|
| `holidays` / festivos | **Decisión explícita**: el entrenamiento 2021-2025 no cubrió los años de forma consistente (backfill recién completado); introduciría ruido estacional confundido con efecto año. Solo se usa en dashboard como anotación. |
| `victimizacion_seguridad` | Varianza ≈ 0 (176 de 1,001 en 0.0 exacto; resto casi todo NULL/cero) → no aporta splits al árbol. |
| `pct_*` actuales (sin `_lag_1`) | **Fuga**: usan el total del mes objetivo dentro de su numerador. |
| Imputación de meses faltantes con 0 | Directiva del proyecto: 0 = cero medido, NULL = sin dato; rellenar fabricaría tasas bajas ficticias y aplastaría el target. |

### 3.6 Diseño de entrenamiento (`xgboots_model.py`)

- **Split temporal** (`train():337-344`): el último año del set de entrenamiento (2025) queda como
  validación → métrica honesta en el tiempo; luego **reentreno final con todo 2021-2025**.
- **Filtros de muestra**: `hay_observacion = 1`, `total_tematica IS NOT NULL`, `pob_dane_2025 > 0`
  → solo meses con dato real entran al ajuste.
- **Categorías alineadas**: `train_categories` se guarda en el fit y se reaplica en `prepare_X`
  con `pd.Categorical` → evita que entrenamiento y pronóstico asignen códigos distintos a los
  clusters.
- **Hiperparámetros**: 400 árboles, profundidad 6, lr 0.05, subsample/colsample 0.9,
  `tree_method="hist"`, `objective="reg:squarederror"` — sesgo-varianza moderado para ~5k filas ×
  48 features.
- **Clip de predicción**: `np.maximum(preds, 0)` → tasas no negativas (XGBoost puede predecir <0
  en la cola baja).
- **Fallback** (`run()`): si el año actual no tiene datos observados → `run_forecast()` (pronóstico
  del mes en curso).
- **Métricas de validación vigentes**: RMSE 14.14, MAE 8.11, R² 0.5928.

### 3.7 Importancia de features (corrida vigente)

| Feature | Importancia |
|---|---:|
| `total_lag_1` | 18.7% |
| `media_movil_3_meses` | 16.8% |
| `art95_num1_...` | 14.5% |
| `pob_dane_2025` | 6.8% |
| `media_movil_6_meses` | 5.3% |
| `log_pob_dane` | 4.5% |
| `media_movil_12_meses` | 3.0% |

Lectura: el modelo vive de la **autorregresión** (lag + medias ≈ 44%), pero `art95` en 3.ª
posición es sospechosa (posible proxy de población → doble contabilidad con `pob_dane`).

---

## 4. GOLD — error, umbrales y consumo

- **`thresholds`**: percentiles **P25/P75 por municipio** sobre 2021-2024 → clasifica
  BAJO/MEDIO/ALTO **relativo a cada territorio** (no un umbral nacional único). Cubre los 1,101
  municipios del forecast (0 sin umbral).
- **`hurtos_personas_consolidado`**: une predicho + real + `error`, `abs_error`, `sq_error`,
  `ape_pct`, `nivel_riesgo_real`, `nivel_riesgo_predicho` → alimenta el histograma de errores y
  las métricas (n = 4,642; MAE 3.04, RMSE 7.27, sesgo +0.26).
- **`hurtos_personas_gold`**: agrega `fecha` (MAKE_DATE), lat/lon, `pres_grup_armados`,
  `pdet_zomac`, `riesgo_victimizacion`, `es_isla` → dashboard AI/BI.

### Diagnóstico de error vigente

| Banda de tasa real | n | Sesgo | MAE |
|---|---:|---:|---:|
| < 10 | 1,993 | **+1.06** (sobreestima) | 1.35 |
| 10–50 | 2,400 | +0.91 | 3.13 |
| 50–100 | 211 | −7.35 | 11.13 |
| 100–200 | 34 | **−30.00** | 31.42 |
| ≥ 200 | 4 | **−127.51** | 127.51 |

- **Regresión a la media**: el modelo "aplasta" los valores altos hacia abajo y los bajos hacia
  arriba. Los 100 casos con mayor error concentran el **71.5% de la suma de errores cuadrados**
  (el RMSE lo sostienen ~2% de los municipios).
- **Impacto en clasificación**: 944 de 4,642 filas (~20%) cambian de nivel de riesgo
  (ALTO 12.4%, MEDIO 24.6%, BAJO 22.9% desclasificados) → degrada la barra apilada de riesgo y
  los mapas del dashboard.

---

## 5. Hallazgos priorizados

| # | Hallazgo | Impacto | Estado |
|---|---|---|---|
| 1 | Silver: 5 columnas del `SELECT` final no existen en ninguna CTE (`tendencia`, `crecimiento_*`, `max_*_anterior`) | **Bloquea** `--step silver` (`UNRESOLVED_COLUMN`) | Plan con código entregado |
| 2 | Forecast: lags desfasados (lag_6/12 → M−7/M−13), `n_obs + 1` fuera de rango (4/7/13) y faltan `historia_*` | Forecast incorrecto / incompleto | Código de reemplazo entregado |
| 3 | Fuga en `crecimiento_*` / `max_*_anterior` (usan el mes actual) | Métricas optimistas + train/serve skew | Plan entregado (redefinir con lags) |
| 4 | Encoding regenerativo vs. `CASE` hardcodeados en Silver | Pivotes silenciosamente incorrectos si cambia el origen | Sin plan |
| 5 | `thresholds` con `COALESCE(total_tematica, 0)` (`consolidated.py:47`) | Meses sin dato entran como 0 → umbrales P25/P50/P75 deprimidos; contradice la directiva "no rellenar con ceros" | Sin plan |
| 6 | `art95` es la 3ª feature más importante sin auditar | Posible proxy de población (doble contabilidad con `pob_dane`) | Auditoría pendiente |
| 7 | `write_predictions_table` escribe por literales VALUES (4,642 filas inline) | Frágil y lento; escala mal | Sin plan |
| 8 | `main.py --step consolidated` ejecuta `run()` + `run_forecast()` → doble corrida | Duplica costo; reescribe forecast 2 veces | Sin plan |
| 9 | Error concentrado en cola alta (subestimación sesgo −30/−127) | 20% de clasificaciones de riesgo erradas | Diagnóstico hecho; fix = vars 3.1–3.3 |
| 10 | `test_query` con `WHERE` después de `ORDER BY`; `WHERE` con precedencia `OR`/`AND` sin paréntesis en forecast | SQL inválido / filtros parciales | Plan entregado |

**Prioridad de atención**: 1 → 2 → 5 → 3 → 4 — los ítems 1 y 2 bloquean la regeneración de
tablas y el 5 distorsiona el panel de riesgo que se va a presentar.

---

## 6. Diagrama de dependencias de datos

```
hurto_personas.parquet (Volumen)
        │ loader.py (rename COLUMNS_MAPPING)
        ▼
hurto_personas ──quality.py──► reporte
        │ bronce.py (encode CASE)
        ▼
hurto_personas_bronce ─────────────────────┐
        │ domains.py STRUCTURE_SILVER_TABLE│
        │   (malla spine + ventanas)       │
        ▼                                  │
hurto_personas_silver                      │ consolidado (ext.)
        │                                  │   │
        ├──── xgboots_model.py ◄───────────┘   │
        │      │                               │
        │      ├─► hurto_personas_prediccion   │
        │      └─► hurto_personas_forecast     │
        │                    │                 │
        ▼                    ▼                 ▼
thresholds ◄── consolidated.py ──► hurtos_personas_consolidado
   (P25/P75 2021-2024)                  │
                                        ▼
                              hurtos_personas_gold ──► Dashboard AI/BI
                              municipios_geo / departamentos_geo
                              holidays (anotación)
```

---

## 7. Próximos pasos recomendados

1. Aplicar fixes de Silver (ítems 1 y 3) y ejecutar `--step silver` para validar el SQL.
2. Reemplazar `build_forecast_query` con la versión de ventanas inclusivas (ítem 2).
3. Corregir `COALESCE(..., 0)` en `generate_threshold_table` (ítem 5) y regenerar `thresholds`.
4. Reentrenar y comparar RMSE/MAE vs. baseline (14.14 / 8.11).
5. Auditoría de `art95` (temporalidad, transformación `art95_tasa`/`art95_log`).
6. Congelar el mapping de encoding (ítem 4).
7. Construir la hoja 3 del dashboard (pronóstico del mes en curso) sobre
   `v_pronostico_mes`.
