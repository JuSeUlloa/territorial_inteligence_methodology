# Diccionario de Datos — Silver y Features del Modelo

> Referencia de las variables que genera la capa **SILVER**
> (`utils/domains.py`) y las que consume el **modelo XGBoost**
> (`models/xgboots_model.py`). Complementa `tecnical-analisis.md`
> (análisis técnico) y `README.md` (operación).
>
> Resumen: **46 features numéricos + 4 categóricos + 2 calendario**,
> target = tasa por 100k habitantes.
> Última actualización: **2026-11-02** (fix de fuga A3 aplicado y
> validado: backtest rolling ├── R² 0.66/0.62/0.61 · MAE 7.25/7.03/6.20).

---

## 0. Convenciones

- **Grano de Silver**: 1 fila = `codigo_dane` (municipio) × `anio` × `mes`.
  Es una **malla completa** (spine): todo municipio tiene fila cada mes
  aunque no haya hurtos.
- **Integridad**: `NULL` = mes sin dato observado; `0` = cero medido.
  Nunca se imputa con ceros (directiva del proyecto).
- **Filtro de entrenamiento**: solo filas con `hay_observacion = 1`
  (o ceros "plausibles": municipios con ≈3 casos/mes — ver
  `LAMBDA_MAX_CERO_PLAUSIBLE`).
- **Fuentes**: columnas marcadas `silver` nacen en las CTEs de
  `utils/domains.py`; marcadas `consolidado` vienen de la tabla
  `consolidado` (estáticas por municipio) y se cruzan con
  `INNER JOIN … ON codigo_dane = codigo_municipio`.

---

## Bloque 1 · Identificación (no entran como features)

| Variable | Tipo | Fuente | Significado |
|---|---|---|---|
| `municipio` | string | silver | Nombre del municipio (DANE) |
| `codigo_dane` | string | silver | Código DANE normalizado a 5 dígitos — **llave** de cruce y de las ventanas |
| `departamento` | string | silver | Nombre del departamento |
| `hay_observacion` | 0/1 | silver | 1 = mes con dato real en la fuente; 0 = fila de la malla sin dato. **Filtro de entrenamiento** |

---

## Bloque 2 · Calendario (features)

| Variable | Fuente | Definición | Por qué entra |
|---|---|---|---|
| `mes` | silver | Mes 1–12 | Estacionalidad; el árbol la captura con splits |
| `anio` | silver | Año 2021+ | Tendencia de largo plazo / efecto año |

---

## Bloque 3 · Autorregresivo temporal (10 features) — la señal dominante

Todas nacen en la CTE `temporal` con
`LAG(total_tematica, n) OVER (PARTITION BY codigo_dane ORDER BY anio, mes)`:

| Variable | Definición |
|---|---|
| `total_lag_1` | Total del mes t−1 (señal más fuerte) |
| `total_lag_2`, `total_lag_3`, `total_lag_6` | Momentum corto/medio |
| `total_lag_12` | t−12 ("mismo mes año anterior"; con huecos lo corrigen `n_obs_*`) |
| `media_movil_3_meses` | `AVG` de 3 meses **gated** (solo si `n_obs_3 = 3` exacto); NULL si huecos |
| `media_movil_6_meses`, `media_movil_12_meses` | ídem con 6 y 12 |

---

## Bloque 4 · Cobertura del histórico (7 features)

| Variable | Definición |
|---|---|
| `n_obs_3` / `n_obs_6` / `n_obs_12` | Meses con dato en cada ventana |
| `historia_meses` | Meses totales con dato del municipio (desde 2021) |
| `historia_insuficiente_3/6/12` | Flag binario: 1 si `n_obs < N` |

---

## Bloque 5 · Momentum / desviación (4 features) — ✅ FIX de fuga aplicado (A3)

| Variable | Definición vigente (sin fuga) | Comentario |
|---|---|---|
| `crecimiento_intermensual` | `(LAG(t,1) / LAG(t,2) − 1) × 100` (NULL si LAG2 ≤ 0) | Antes usaba el target (fuga). Ahora mide "cómo veníamos" |
| `crecimiento_interanual` | `(LAG(t,1) / LAG(t,12) − 1) × 100` | ídem |
| `max_mes_anterior` | `LAG(t,1) − LAG(t,2)` | Δ del mes anterior |
| `max_anio_anterior` | `LAG(t,1) − LAG(t,12)` | ídem; se corrigió además el off-by-one del forecast (usaba t−13) — hallazgo A2 |

> **Cierre del hallazgo A3** (2026-11-02): train y forecast usan la misma
> fórmula rezagada. Efecto medido en el backtest rolling: las métricas
> anteriores (R² 0.98, MAE 0.59-1.03) eran espejo de la fuga; el honesto es
> R² 0.61-0.66 y MAE 6.2-7.2, con `gap` de sobreajuste 1.1-1.5x (antes
> 2.43x). Solve del fold 2025 (hold-out pendiente del modelo): ✅ OK.
> Pendiente activo: deriva de sesgo por año (folds 2023/2024) — se resuelve
> con datos, no con el modelo (ver `tecnical-analisis.md` §8.5b).

---

## Bloque 6 · Composición del hecho (9 features, todas rezagadas)

`pct_x = categoría / total × 100` (solo si `total > 0`) → `LAG(pct_x, 1)`:

| Variable | Medida (lag t−1) |
|---|---|
| `pct_zona_urbana_lag_1` / `pct_zona_rural_lag_1` | Composición territorial |
| `pct_arma_fuego/blanca/contundente_lag_1` | Modus operandi |
| `pct_genero_masculino/femenino_lag_1` | Perfil de víctima |
| `pct_edad_adultos/jovenes/menores_lag_1` | Segmento etario |

> Las versiones sin `_lag_1` están **excluidas** por fuga directa.

---

## Bloque 7 · Estructurales de `consolidado` (estáticas por municipio)

`JOIN consolidado c ON a.codigo_dane = c.codigo_municipio`:

| Variable | Significado | Nota |
|---|---|---|
| `codigo_departamento` | Código DANE del departamento | Efecto fijo regional |
| `pob_dane_2025` | Población proyección DANE 2025 | **Denominador del target** |
| `log_pob_dane` | `LOG1P(pob_dane_2025)` | Comprime cola asimétrica |
| `pdet`, `zomac`, `pdet_zomac` | PDET / ZOMAC / cruce | Contexto estructural |
| `riesgo_victimizacion` | Índice 0–1 | Riesgo de victimización |
| `pres_grup_armados` | Presencia de grupos armados | crimen + sesgo de denuncia |
| `victimizacion_seguridad` | Índice por seguridad | Varianza ≈ 0 |
| `pib2024_peso_relativo_mun_valor_agregado_departamental` | Peso del PIB | Contexto económico |
| `art95_num1_base_datos_negativa_del_articulo_106_ley_1453_2011` | BD negativa rural | ⚠️ posible proxy de población — auditoría pendiente (A6) |
| `longitud`, `latitud` | Coordenadas | Proxy espacial |
| `cluster_rv` / `cluster_pga` / `cluster_vs` | **Categóricas** (ordinales Bajo→Alto) | `enable_categorical=True`; `train_categories` se reaplican en `prepare_X` para evitar skew |
| `municipio_isla_area_no_municipalizada` | Binaria | Turismo/aislamiento |

---

## Bloque 8 · Target

| Etapa | Definición |
|---|---|
| `target` (train) | `(COALESCE(total_tematica, 0) / pob_dane_2025) × 100000` — tasa por 100k |
| Fit | XGBoost plain (`reg:squarederror`, 400 árboles, depth 6, lr 0.05) con target crudo |
| `predict` | `max(pred, 0)` — clip no negativo |
| Persistencia | `save_model()`: pkl directo (local + volumen UC vía `ModelStore`) |

> Estado 2026-11-02: los ajustes de nivel probados ese día (target log1p,
> sample_weight, caps por bin, calibrador por factor) **se retiraron** —
> eran parche por la deriva de nivel anual, cuya solución de raíz es datos
> (covariables anuales + historia continua; ver `tecnical-analisis.md`
> §8.5b). Quedan documentados para reactivar si se decide.

---

## Bloque 9 · Excluidas del modelo (y por qué)

| Variable | Razón |
|---|---|
| `holidays` / `n_festivos` | Solo dashboard (decisión explícita) |
| `pct_*` sin `_lag_1` | Fuga |
| Imputación con 0 | Directiva: 0 = cero medido |
| `total_lag_4`, `total_lag_5` | Generadas en silver, no incluidas en `NUM_VARS_SILVER` |

---

## Bloque 10 · Uso en el carril barrio (fase B, piloto Buenaventura)

`models/municipality_test.py::generate_training_municipio()` reusa los
bloques 2–5 + target re-escalado con población por barrio (la tabla
`poblacion_municipio` hoy es ancha `pob_2020..pob_2024` — se toma la más
reciente con COALESCE). Carga el mismo `.pkl` nacional. Gates propios:
`spearman ≥ 0.35` y `mae_rel ≤ 0.25` — si no pasan, **no se publica**
(estado 2026-11-02: REPROBADOS con el modelo honesto, piloto pendiente de
feed 2025-2026 o modelo barrial propio; ver `tecnical-analisis.md` B7).

---

## Apéndice · SQL de referencia de las ventanas

```sql
WINDOW w AS (PARTITION BY codigo_dane ORDER BY anio, mes)

LAG(total_tematica, 1)  OVER w AS total_lag_1,   -- … hasta 12

-- cobertura y gate de medias móviles
COUNT(total_tematica) OVER (… ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS n_obs_3
CASE WHEN n_obs_3 = 3 THEN AVG(total_tematica) OVER … END AS media_movil_3_meses

-- momentum corregido (sin fuga, con datos ≤ t−1)
(LAG(total_mes,1) OVER w / NULLIF(LAG(total_mes,2) OVER w, 0) - 1) * 100  AS crecimiento_intermensual
```

Mantenimiento: cualquier cambio de features se refleja aquí y en
`tecnical-analisis.md` §3.
