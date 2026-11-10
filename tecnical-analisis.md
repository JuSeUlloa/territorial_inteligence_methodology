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
| `bronze_municipio` (SQL directo) | CTAS documentado en README | `hurto_personas_bronce_municipio` (3,531 eventos) | 1 fila = **evento** homologado a la estructura del bronce nacional + `id_barrio`/`barrio` (carril independiente, no pasa por `BronzeEncoder`) |

Entorno: Databricks (`workspace.territorial_inteligence`), CLI v1.14.1, Python 3.12, uv.

Tablas de apoyo: `holidays` (2021-2026, 18 festivos/año), `municipios_geo` (1,152),
`departamentos_geo` (33), `consolidado` (1,120 municipios con variables estructurales).
Carril municipio: `hurto_personas_municipio` (feed barrial crudo), `poblacion_municipio`
(75 barrios), `municipio_barrios_geo` (203 polígonos) — ver README, sección "Carril Municipio".

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

---

## 8. Estado del flujo — Dos fases (implementado vs. pendiente)

> Resumen de estado: qué está implementado, qué queda pendiente, y cómo se
> relacionan las **dos fases** del pipeline. Las secciones 1–7 de este documento
> cubren el detalle técnico de la fase A; esta sección consolida ambas fases
> y sus pendientes.

### 8.0 Visión general

El proceso tiene **dos fases con granularidad distinta** que comparten catálogo,
schema, SQL warehouse y el job DAB `territorial_inteligence_pipeline`:

```
┌──────────────────────────── FASE A · NACIONAL ────────────────────────────┐
│  Granularidad: MUNICIPIO × MES (Colombia completa, 1.100+ municipios)     │
│                                                                          │
│  quality ► encode_bronze ► transform_silver ► train_model ► consolidated │
│         (cadena en resources/territorial_intelligence.job.yml)            │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │ (consolidated_gold dispara)
┌────────────────── FASE B · CARRIL BARRIO (PILOTO) ────────────────────────┐
│  Granularidad: BARRIO × MES (piloto municipal 76109 · Buenaventura)       │
│                                                                          │
│  municipality_carril (depende de consolidated_gold)                      │
│  Disagrega la predicción municipal al barrio: el modelo decide CUÁNTO,    │
│  los shares históricos deciden DÓNDE — no hay modelo ML barrial propio.  │
└───────────────────────────────────────────────────────────────────────────┘
```

**Regla operativa**: las 6 tasks corren en cadena. Si `municipality_carril`
falla (o sus gates reprueban), se skipean las tasks posteriores, pero la fase
A ya quedó publicada — no se pierde trabajo.

| | Fase A (Nacional) | Fase B (Piloto barrio) |
|---|---|---|
| Granularidad | municipio × mes | barrio × mes |
| Alcance | Colombia (1,120 municipios) | 76109 Buenaventura (75 barrios con población) |
| Modelo | XGBoost sobre tasas (48 features) | Reusa el `.pkl` nacional (`ModelStore`) |
| Target | `tasa_real` /100k hab | conteo disagregado → tasa por barrio |
| Periodo entrenamiento | 2021-2025 | 2020-2024 (feed barrial) |
| Salidas gold | `hurtos_personas_gold` | `hurto_personas_gold_municipio` |
| Dashboard | AI/BI hojas 1-2 | **pendiente** (mapa/ficha por barrio) |

### 8.1 Fase A — cadena de tasks y métricas vigentes

| # | Task (`--step`) | Código | Salida | Estado |
|---|---|---|---|---|
| 1 | `quality` | `bronze/quality.py` | reporte en stdout (no bloqueante) | ✅ |
| 2 | `encode_bronze` | `bronze/bronce.py` | `hurto_personas_bronce` (735,802 eventos) | ✅ |
| 3 | `transform_silver` | `silver/silver.py` + `utils/domains.py` | `hurto_personas_silver` (malla municipio × mes, lag/medias gated) | ✅ |
| 4 | `train_model` | `models/xgboots_model.py` | `hurto_personas_prediccion` + `hurto_personas_forecast`; pkl → `ModelStore` (volumen UC `files/models/xgboost_hurtos.pkl`) | ✅ |
| 5 | `consolidated_gold` | `gold/consolidated.py` | `thresholds` (P25/P75 con shrinkage bayesiano), `hurtos_personas_consolidado`, `hurtos_personas_gold` | ✅ |

Métricas del run 2026-10-28: validación 2025 **RMSE 2.31 · MAE 0.60 · R² 0.98**;
forecast 2026-10 para 1.100 municipios; `hurto_personas_consolidado` 9.908 filas.
Features dominantes: `media_movil_3_meses` (40%), `max_mes_anterior` (12%),
`total_lag_1` (11%).

**Pendientes de fase A** — son los hallazgos 1–10 de la sección 5 de este
documento, más estos adicionales:

| # | Pendiente | Dónde | Impacto | Estado |
|---|---|---|---|---|
| A1–A10 | Hallazgos 1–10 de la **sección 5** (Silver UNRESOLVED, forecast con ventanas incorrectas, fuga en `crecimiento_*`, encoding regenerativo, COALESCE en `thresholds`, auditoría de `art95`, writes con VALUES inline, doble corrida de forecast, cola alta del error, SQL inválido en queries) | ver sección 5 | ver sección 5 | **A3 (fuga) CERRADO 2026-11-02**: train/forecast con fórmulas solo-rezago + off-by-one del forecast corregido. Backtest final (solo-fix-A3): R² 0.66/0.62/0.61 · MAE 7.25/7.03/6.20 · gap 1.52/1.31/1.12 · **2025 ✅**, 2023/2024 por deriva de sesgo por año. Evidencia "más datos → mejores métricas" en §8.5b (MAE −9.6%, sesgo ×3 con +1 año de train). Los ajustes de nivel (weight/caps/calibrador) se probaron y **se retiraron** (parche por la deriva — mejor resuelto con covariables anuales, §8.5b). Resto: ver sección 5 |
| A11 | `ModelStore.load()` leía `resp.content` (atributo inexistente del SDK → stream vacío → `EOFError`); corregido a `resp.contents.read()`. **Pendiente**: re-verificar en target `prod` | `utils/model_store.py` | Carga del modelo | ✅ Fix aplicado |
| A12 | **Hoja 3 del dashboard** (pronóstico del mes en curso) sobre `v_pronostico_mes` | Dashboard AI/BI | Falta construir | ⏳ Pendiente |
| A13 | Manifest del `.pkl` (versión/fecha/métricas del train) — hoy es un binario opaco en volumen | `ModelStore` + training job | Trazabilidad de modelo | ⏳ Sin plan |

### 8.2 Fase B — carril barrio (piloto Buenaventura)

La fase B se alimenta de un **feed crudo barrial** (`hurto_personas_municipio`)
que **no** pasa por `BronzeEncoder`: se homologa con un CTAS SQL al léxico del
bronce nacional (reglas y SQL en `README.md`, sección "Carril Municipio"). Los
eventos se agregan en la SILVER barrial (`hurto_personas_silver_municipio`,
barrio × mes) y el task `municipality_carril` ejecuta la cadena de publicación.

Código: `models/municipality_test.py` (`MunicipalityModel.run()`).

| # | Paso | Qué hace | Tabla in/out | Estado |
|---|---|---|---|---|
| B1 | `generate_training_municipio()` | CTAS desde SILVER barrial + contexto del consolidado nacional; ventana hist 2020-2024 | → `hurto_personas_training_municipio` (4.500 filas · 75 barrios · 2020-2024) | ✅ |
| B2 | `generate_shrinkage_thresholds()` | P25/P50/P75 por barrio 2020-2024 con shrinkage bayesiano (publica `threshold_type`, `shrinkage_factor`) | → `thresholds_shrinkage_municipio` | ✅ |
| B3 | `score_data()` | XGBoost **nacional** reusado (`ModelStore.load()`) → `conteo_predicho` + `tasa_predicha` por barrio | — | ✅ |
| B4 | `validate_scores()` + `evaluate_folds()` | Backtest rolling por año (2020-2023 gate, 2024 solo referencia); Spearman por fold, `mae_mensual_rel` | — | ✅ (PASS vigente: spearman mediana 0.383 · mae_rel 0.232) |
| B5 | `check_gates()` | **Gates de publicación**: `GATE_SPEARMAN_MIN = 0.35` · `GATE_MAE_REL_MAX = 0.25`. Si se reproban → `ValueError` y **no se publica nada** | — | ✅ |
| B6 | `write_predictions_table()` | Histórico del año de publicación | → `hurto_personas_prediccion_municipio` | ✅ |
| B7 | `score_forecast()` + `write_forecast_table()` | Pronóstico del mes en curso (2026-10) | → `hurto_personas_forecast_municipio` (75 filas) | ✅ |
| B8 | `consolidar_predicciones()` | Forecast + histórico | → `hurto_personas_prediccion_municipio` (975 filas) | ✅ |
| B9 | `generate_gold_municipio()` | Gold: tasas real/predicha, error, nivel de riesgo (P25/P50/P75 del barrio), contexto (`pres_grup_armados`, `pdet_zomac`, `riesgo_victimizacion`, es isla) | → `hurto_personas_gold_municipio` (975 filas · niveles: ALTO 88 · MEDIO 711 · BAJO 101 · 75 sin tasa) | ✅ |

> ⚠️ **Nota operativa**: el `.pkl` lo sube al volumen el task `train_model` de
> la **fase A**. La cadena obligatoria del job lo garantiza dentro de un run,
> pero al correr `--step municipality` aislado el pkl puede quedar
> desactualizado respecto a los datos.

**Pendientes de fase B**:

| # | Pendiente | Dónde | Impacto | Estado |
|---|---|---|---|---|
| B1 | **CTAS del bronce barrial sin código mantenido** (`hurto_personas_bronce_municipio`): vive como SQL en el editor de Databricks, no en el repo ni en el job → si se pierde ese SQL, la fase B no es reproducible | `README.md` §"Creación de la tabla" | Reproducibilidad de la fase B | ⏳ Priorizar: versionarlo en `src/scripts/` o como task previa del job |
| B2 | Feed barrial **solo 2020-2024** (sin 2025-2026): toda predicción disagregada arrastra ese desfase, hoy expuesto vía `meses_desfase` | `hurto_personas_municipio` (fuente) | Predicción barrial desactualizada; gates podrían verse afectados con feed nuevo | ⏳ Gestionar actualización del feed |
| B3 | Cobertura de población: solo el **35.8%** de eventos del feed cruza con `poblacion_municipio` → 75 de 203 barrios tienen tasa; los micro-barrios sin población quedan fuera del gold | `poblacion_municipio` | Mapa/ficha incompleta del piloto | ⏳ Decidir: agregar población faltante vs. dejar fuera documentado |
| B4 | Gates reprobados ⇒ no se publica nada (`ValueError`): **no hay alerta/notificación** al equipo | `check_gates()` + job config | Mala visibilidad del fallo | ⏳ Sugerido: notificaciones por email en el job DAB |
| B5 | `municipality_carril` reusa el `.pkl` nacional sin validar features ni calibrar a nivel barrio (modelo entrenado sobre tasas municipales, usado sobre conteos barriales) | `MunicipalityModel.score_data()` | Riesgo metodológico del piloto | ⏳ Documentar supuestos + validar antes de producción |
| B6 | `poblacion_municipio` **cambió de esquema** (formato ancho `pob_2020..pob_2024`) y rompió el SQL del carril; adaptado con COALESCE por año el 2026-11-02. Falta fijar el contrato de esquema de la tabla | `generate_training_municipio` / `score_forecast` / shrinkage | Frágil ante cambios futuros de la fuente de población | ✅ Fix aplicado · ⏳ contrato de esquema |
| B7 | **Gates del carril reprobaron** en los runs 2026-11-02 con el modelo nacional honesto (mejor fold: spearman 0.393, `mae_rel` 0.368 > 0.25) → el carril no publica y las tablas de fase B quedan con los datos del run de octubre (stale). Causa raíz: el pkl nacional cambió (sin fuga) y deja de explotar el patrón barrial que la fuga le daba gratis | `check_gates()` | El piloto requiere feed 2025-2026 (B2) o modelo barrial propio para volver a publicar | ⏳ Bloquea publicaciones nuevas del piloto |

### 8.3 Pendientes transversales (afectan ambas fases)

| # | Pendiente | Impacto |
|---|---|---|
| T1 | Job sin `max_retries` ni notificaciones en `resources/territorial_intelligence.job.yml`: un fallo transitorio del warehouse aborta la cadena | Reejecución manual; fallas invisibles (ver B4) |
| T2 | Escalar el piloto a más municipios: la fase B es dinámica, pero `id_barrio` = `MD5(nombre)` colisiona entre municipios → extender la llave a `CONCAT(codigo_dane, '-', id_barrio)` en todas las tablas del carril antes de sumar un segundo municipio | Bloquea escalar fase B |
| T3 | Congelar el mapping de encoding en una tabla de dimensión (fase A) y re-homologar el carril (fase B) con la misma tabla | Consistencia entre fases |
| T4 | Runbook de operación: no hay sección de rollback ni de qué hacer si `municipality_carril` reproba gates | Operación |
| T5 | Tests unitarios de los gates y de la homologación del feed barrial (códigos 0, léxico, buckets de edad) | Calidad del release |
| T6 | Unificar documentación de parámetros/targets del job (hoy repartido en README, AGENTS.md, `databricks.yml`) | Operación |

### 8.4 Orden de ejecución recomendado

1. **A1 — Regeneración de Silver** (bloqueante, ver sección 5).
2. **B1 — Versionar el CTAS barrial en el repo/job** (reproducibilidad del piloto).
3. **A5 — Fix de `thresholds` COALESCE → P25/P75 sobre meses observados**.
4. **A2/A3 — Forecast y fuga de `crecimiento_*`** (reentrenar y comparar métricas).
5. **T2 — Escalar piloto**: llave compuesta contra colisión de `id_barrio`; sumar municipios al feed.
6. **Dashboards**: hoja 3 de fase A (pronóstico del mes) y mapa/ficha barrial de fase B.

### 8.5 Referencia rápida de ejecución

```bash
# Fase A + B completas (job DAB, target dev)
databricks bundle deploy -t dev --profile DEFAULT
databricks bundle run territorial_inteligence_pipeline -t dev --profile DEFAULT

# Un paso individual (wheel instalado localmente)
python -m territorial_inteligence_methodology.main --step silver
python -m territorial_inteligence_methodology.main --step municipality
```

Parámetros del job: `catalog` y `schema` (variables DAB; dev/prod hoy:
`workspace.territorial_inteligence`).

### 8.6 API de consulta sobre Databricks

API REST **Flask** en `src/territorial_inteligence_methodology/app/` que expone
las salidas gold de las dos fases (y las geometrías) sobre Databricks: cada
endpoint arma SQL y lo ejecuta vía `SqlExecutor` → `execute_statement` contra
el **SQL warehouse** (`workspace.territorial_inteligence`). No ataca directamente
los volúmenes ni clusters.

```
cliente ─► /api/territorial_intelligence/* ─► routes.py (Blueprint)
                                              └► controller.py (status codes, 400/503)
                                                  └► services.py (SQL por tabla)
                                                      └► SqlExecutor → warehouse
```

| Método + ruta | Fuente | Qué devuelve |
|---|---|---|
| GET `/gold` | `hurtos_personas_consolidado` (fase A) | histórico pred/real por municipio × mes (excluye el último mes) |
| GET `/gold/<mes>` | ídem | ídem filtrado por mes |
| GET `/prediction` | consolidado (fase A) | predicción del último mes con conteo estimado |
| GET `/prediction/neighborhood` | `hurto_personas_gold_municipio` (fase B) | gold barrial completo (2020-2024 + forecast) — piloto Buenaventura, 975 filas |
| GET `/prediction/neighborhood/current` | ídem | solo el máximo año (forecast del mes en curso, 75 filas) |
| GET `/prediction/neighborhood/<mes>` | ídem | gold barrial filtrado por mes |
| GET `/department` | `departamentos_geo` | GeoJSON departamental (con `ST_Simplify`) |
| GET `/neighborhood` | `municipio_barrios_geo` × feed crudo | GeoJSON de barrios del carril |
| GET `/department/<cod>/municipality` | `municipios_geo` | GeoJSON de los municipios de un departamento |
| GET `/municipality/<codigo_dane>` | `municipios_geo` | GeoJSON de un municipio (valida 5 dígitos DANE → 400 si no) |

Errores: respuesta JSON `{status, error}` — `400` validación (código inválido),
`503 degraded` si la query al warehouse falla, `500` para excepciones no
manejadas; CORS abierto para `/api/*`.

**Arranque** (smoke-tested contra `workspace.territorial_inteligence`,
run 2026-10-28 — los 10 endpoints respondieron 200):

```bash
uv sync                          # deps ya incluyen flask, flask-cors, gunicorn
uv run --frozen python src/scripts/run_api.py          # gunicorn (producción API_DEBUG no definido)
API_DEBUG=true uv run python src/scripts/run_api.py    # dev server
# o entrypoint del wheel:
uv run --frozen api              # pyproject [project.scripts] api = app:main
curl -s localhost:3300/api/territorial_intelligence/prediction/neighborhood/current
```

Puerto: `$PORT` (default **3300**, unificado; Databricks Apps inyecta `PORT`).

**Respecto al despliegue sobre Databricks, aún NO está desplegada**:

- La task `api` existe en `main.py` pero **no** está en
  `resources/territorial_intelligence.job.yml` (el job termina en
  `municipality_carril`) — no la agregamos porque un server Flask lanzado como
  task de job tendría ciclo de vida del cluster (se rompería al terminar el run).
- Camino previsto: **Databricks Apps** (source code con la app Flask, comando de
  arranque `run_api.py`, auth vía OAuth del service principal del app;
  declarada como recurso `apps` en los DABs con `$CATALOG`/`$SCHEMA` como env
  del app). Queda pendiente su definición en `resources/` y smoke-test en dev.
- Auth: en local el `WorkspaceClient` usa las credenciales del desarrollador;
  al desplegar debe usar la identidad del service principal del app.

**Despliegue como Databricks App (hecho el 2026-10-28)**:

- App creada/apuntando en el workspace: **`territorial-api`**
  → `https://territorial-api-7474655446021466.aws.databricksapps.com`
  (service principal propio del app: `47944e34-28bc-4de6-b9b7-75b6b2e69cfa`).
- Source code de la app: `apps/territorial-api/` — runtime mínimo (flask,
  flask-cors, gunicorn, databricks-sdk, python-dotenv vía `requirements.txt`;
  Databricks Apps **no** soporta `pyproject.toml`/`uv.lock`). El paquete
  `src/territorial_inteligence_methodology` se copia a
  `apps/territorial-api/src/` con `bash src/scripts/build_api_app.sh`
  (directorio `.gitignore`-rado, generado).
- `app.yaml` define el comando (`python app_server.py`) e inyecta
  `DATABRICKS_WAREHOUSE_ID=56a977392d922752`; host `0.0.0.0` y puerto
  `DATABRICKS_APP_PORT` (reglas de la plataforma: ver arriba).
- Permisos UC otorgados al SP del app: `USE CATALOG workspace`,
  `USE SCHEMA workspace.territorial_inteligence` y `SELECT` por tabla
  (24 tablas). Pendiente: `GRANT SELECT ... FUTURE TABLES` no se aplicó —
  así que al crear tablas nuevas hay que re-otorgar.
- Despliegue en modo `SNAPSHOT` (`SNAPSHOT mode`):

  ```bash
  # 1. sincronizar el paquete
  bash src/scripts/build_api_app.sh
  # 2. subir al workspace
  rm -rf /tmp/api-src && mkdir -p /tmp/api-src
  rsync -a apps/territorial-api/ /tmp/api-src/
  databricks workspace import-dir /tmp/api-src \
    /Workspace/Users/juan.ulloa@usantotomas.edu.co/territorial-api-src \
    --overwrite --profile DEFAULT
  # 3. desplegar (valida, aplica config y arranca el app)
  databricks apps deploy territorial-api --auto-approve --mode SNAPSHOT \
    --source-code-path /Workspace/Users/juan.ulloa@usantotomas.edu.co/territorial-api-src \
    --profile DEFAULT
  ```

- **Smoke test en dev (2026-10-28, 9/9 rutas)**: con el token de usuario del
  desarrollador (`databricks auth token --profile DEFAULT`):
  `/gold` · `/gold/10` · `/prediction` · `/prediction/neighborhood` ·
  `/prediction/neighborhood/10` · `/prediction/neighborhood/current`
  (75 barrios del forecast) · `/department` · `/neighborhood` ·
  `/department/91/municipality` · `/municipality/76109` → **200 OK**;
  `/municipality/761` (inválido) → **400**. Sin token de sesión del app,
  el proxy responde 401 (es el comportamiento de Databricks Apps:
  requieren usuario autenticado del workspace con acceso al app).

### API pública para Angular/Vercel (opción 1 implementada 2026-10-28)

Ver `api-public/README.md`: proxy serverless en Vercel (`api/dbx.mjs`) que va
**directo al SQL warehouse** con un SP dedicado de OAuth M2M
(`territorial-api-public`, `f9ae9d7b-…`, secret expira 2028-10-27) — así el
frontend público no depende del login de Databricks Apps. Grants UC otorgados
al nuevo SP (USE CATALOG/SCHEMA + SELECT en las 24 tablas). Validado
end-to-end con Statement Execution API.

**Pendientes de la API**:

| # | Pendiente | Detalle |
|---|---|---|
| API1 | ~~Despliegue como Databricks App~~ → **HECHO 2026-10-28**: app `territorial-api` RUNNING (ver arriba); pendiente recurso `apps` declarado en DABs para despliegue declarativo por bundle | hecho |
| API2 | Auth de servicio: el app ya usa su SP auto-inyectado del runtime (OAuth M2M) via `WorkspaceClient()`; no requiere credenciales de desarrollador ✅ | hecho |
| API3 | TTL declarado pero **sin usar**: `Controller(ttl_seconds=600)` — implementar caché (o quitar el parámetro) | `controller.py` / `services.py` |
| API4 | Paginación/truncamiento: `SqlExecutor` exige resultado inline completo (raise si excede el bloque); endpoints voluminosos (`/gold`, `/neighborhood` GeoJSON) pueden romper al crecer el universo | `utils/sql_executor.py:37` |
| API5 | Smoke-test automatizado (hoy fue manual con curl); pytest de los 10 endpoints | `tests/` |
| API6 | Rate limit / auth de API para exposición externa (hoy CORS abierto, autorización usuario de Databricks Apps) | seguridad |
| API7 | `WEB_CONCURRENCY` del runtime Flask dev server (DrNB): cuando el tráfico de producción suba, migrar el command del `app.yaml` al wrapper `run_api_gunicorn.py` | capacidad |
