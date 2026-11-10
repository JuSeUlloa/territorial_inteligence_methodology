# Territorial Intelligence Methodology

Pipeline de datos para análisis territorial de hurtos en Colombia. Implementa la arquitectura **medallion** (bronze → silver → gold) en Databricks, con un modelo predictivo XGBoost que estima la tasa de hurtos por cada 100,000 habitantes por municipio y mes.

## Descripción del Proyecto

El pipeline procesa datos de hurtos contra personas, enriquece la información con variables socioeconómicas y geográficas de los municipios colombianos, y genera predicciones de la tasa de hurtos por municipio y mes. La salida gold consolida tasas reales y predichas, errores, niveles de riesgo y geometrías para consumo analítico.

**Target (variable objetivo):**

```
tasa_hurtos = (total_tematica / pob_dane_2025) * 100000
```

Donde `total_tematica` es el conteo de hurtos por municipio/mes y `pob_dane_2025` es la población proyectada por el DANE para 2025.

## Arquitectura del Pipeline

El proyecto sigue el patrón **medallion** de Databricks, organizado en tres capas:

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ BRONZE  │────▶│ SILVER  │────▶│  GOLD   │
│ (raw)   │     │(curated)│     │(predict)│
└─────────┘     └─────────┘     └─────────┘
  Parquet         Agregación       XGBoost
  → Delta         Ventanas         Predicciones
                  móviles          2025
```

### Bronze (Capa Cruda)

**Ubicación:** `src/territorial_inteligence_methodology/bronze/`

Convierte archivos Parquet en tablas Delta dentro de Unity Catalog. Incluye:

| Módulo | Clase | Función |
|--------|-------|---------|
| `loader.py` | `ParquetLoader` | Carga archivos Parquet a volumes de Databricks |
| `bronce.py` | `BronzeEncoder` | Codifica variables categóricas (género, zona, armas, edad) |
| `quality.py` | `Quality` | Validación de datos (nulos, duplicados, estadísticas) |
| `schema.py` | `SchemaGenerator` | Mapeo de columnas originales a nombres normalizados |

### Silver (Capa Curada)

**Ubicación:** `src/territorial_inteligence_methodology/silver/silver.py`

Agrega los datos por municipio/mes/año y genera features temporales:

- **Porcentajes de composición:** zona urbana/rural, armas/blancas/contundentes, género, edad, día de la semana
- **Medias móviles:** 3 y 6 meses
- **Tendencia:** regresión lineal en ventana de 6 meses
- **Crecimiento:** intermensual e interanual
- **Lags:** mes anterior, máximo del año anterior

### Gold (Capa de Predicción)

**Ubicación:** `src/territorial_inteligence_methodology/models/xgboots_model.py` y `src/territorial_inteligence_methodology/gold/consolidated.py`

El modelo XGBoost genera las predicciones. Después, la consolidación gold combina las predicciones con las tasas reales, población, umbrales percentiles, métricas de error y clasificación de riesgo. Detalles en la sección [Modelo Predictivo](#modelo-predictivo).

La tabla gold está diseñada para alimentar el dashboard AI/BI y los análisis geográficos a nivel departamental y municipal.

## Modelo Predictivo

### Algoritmo

**XGBoost Regressor** (extreme gradient boosting) con los siguientes hiperparámetros:

```python
XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)
```

### Features

El modelo utiliza 44 features de tres fuentes:

**1. Composición del hurto (22 variables):** Porcentuales de distribución por zona, arma, género, edad y día de la semana.

**2. Temporales (8 variables):** Medias móviles, tendencia, crecimiento intermensual/interanual, y lags del mes anterior.

**3. Variables socioeconómicas de consolidado (14 variables):** Población, PDET, ZOMAC, riesgo de victimización, presencia de grupos armados, PIB, latitud/longitud, entre otras.

### Advertencia: Fuga de Datos (Data Leakage)

Algunas features temporales (media móvil, tendencia, crecimiento) incluyen el mes actual (`CURRENT ROW`), lo que puede introducir fuga de datos si `total_tematica` se usa para calcular el target. Para un pronóstico más limpio, considerar excluir o recalcular con desplazamiento temporal.

### Entrenamiento

- **Train:** 2021-2024 (con JOIN a tabla `consolidado` para variables socioeconómicas)
- **Validación:** Holdout temporal 2024 (métricas honestas de generalización)
- **Reentrenamiento:** Todo el dataset 2021-2024 para el modelo final
- **Predicción:** 2025 (sin target, solo variables de silver + consolidado)

### Métricas

El modelo reporta:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coeficiente de determinación)
- **Feature Importance** (top 15 variables)

### Salidas

1. **Modelo entrenado:** `models/xgboost_hurtos.pkl` (joblib)
2. **Tabla Delta:** `{catalog}.{schema}.hurto_personas_prediccion` con predicciones de 2025
3. **Métricas en consola:** RMSE, MAE, R² y feature importance

## Controles de Calidad

`bronze/quality.py` ejecuta validaciones configurables sobre las tablas del pipeline:

| Validación | Descripción |
|------------|-------------|
| Conteo de filas | Verifica el volumen de cada tabla |
| Valores nulos | Calcula nulos por columna y compara umbrales |
| Duplicados | Valida duplicados usando claves de negocio |
| Integridad referencial | Detecta códigos municipales que no existen en `consolidado` |
| Umbrales | Reporta violaciones de calidad configuradas |

Ejecutar las validaciones:

```bash
uv run python -m territorial_inteligence_methodology.main --step quality
```

## Estructura del Proyecto

Solo se incluyen archivos tracked (no excluidos por `.gitignore`). Archivos generados (`.venv/`, `__pycache__/`, `build/`, `dist/`, `*.egg-info`, `*.pkl`), datos locales (`data/`), configuración secreta (`.env`), DABs (`resources/`) e IDE (`.vscode/`) se omiten.

```
territorial_inteligence_methodology/
├── .gitignore
├── AGENTS.md                                # Instrucciones para agentes de IA
├── CLAUDE.md                                # Configuración de Claude
├── README.md                                # Este archivo
├── pyproject.toml                           # Dependencias y configuración del proyecto
├── uv.lock                                  # Versiones bloqueadas por uv
├── databricks.yml                           # Declarative Automation Bundles
├── fixtures/
│   └── .gitkeep
├── src/
│   ├── territorial_inteligence_methodology/
│   │   ├── __init__.py
│   │   ├── main.py                          # Orquestador del pipeline (argparse --step)
│   │   ├── bronze/
│   │   │   ├── bronce.py                    # Codificación de variables
│   │   │   ├── config.py                    # Configuración (catalog, schema, tablas)
│   │   │   ├── loader.py                    # Carga Parquet → Volume → Delta
│   │   │   ├── quality.py                   # Validación de calidad
│   │   │   └── schema.py                    # Mapeo de columnas
│   │   ├── silver/
│   │   │   └── silver.py                    # Agregación y features temporales
│   │   ├── gold/
│   │   │   └── consolidated.py               # Consolidación de predicciones, tasas y riesgo
│   │   ├── models/
│   │   │   └── xgboots_model.py             # Modelo XGBoost
│   │   └── utils/
│   │       ├── client_api.py                # Cliente de APIs externas
│   │       ├── databricks_client.py         # Cliente singleton de Databricks
│   │       ├── domains.py                   # Mapeo de columnas originales
│   │       ├── histograma_error.py           # Análisis de errores del modelo
│   │       └── sql_executor.py              # Ejecución de queries SQL
│   ├── territorial_inteligence_methodology_etl/
│   │   ├── README.md
│   │   └── transformations/
│   │       ├── sample_trips_territorial_inteligence_methodology.py
│   │       └── sample_zones_territorial_inteligence_methodology.py
│   └── scripts/
│       ├── load_data.py                     # Carga de archivos locales
│       └── load_holidays_data.py            # Carga de festivos desde API
└── tests/
    ├── conftest.py                          # Configuración de pytest + Databricks Connect
    └── sample_taxis_test.py                 # Test de ejemplo
```

## Instalación

### Requisitos

- Python 3.12+
- uv (gestor de paquetes): https://docs.astral.sh/uv/getting-started/installation/
- Databricks CLI (para despliegue): https://docs.databricks.com/dev-tools/cli/install
- macOS: `brew install libomp` (requerido por XGBoost)

### Configurar entorno local

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd territorial_inteligence_methodology

# Instalar dependencias
uv sync --dev

# Configurar variables de entorno (editar .env)
CATALOG=workspace
SCHEMA=territorial_inteligence
VOLUME_NAME=hurto_personas
TABLE_NAME=hurto_personas
BRONZE_TABLE_NAME=hurto_personas_bronce
SILVER_TABLE_NAME=hurtos_personas_silver
```

### Solución de problemas en macOS

Si al importar `xgboost` aparece el error:

```
Library not loaded: @rpath/libomp.dylib
```

Ejecutar:

```bash
brew install libomp
```

Esto instala el runtime OpenMP en `/opt/homebrew/opt/libomp/lib/libomp.dylib`, que XGBoost busca por defecto en Apple Silicon.

## Umbrales de Riesgo

La tabla `{catalog}.{schema}.thresholds` se genera a partir de las tasas históricas de 2021-2024. Contiene umbrales percentiles por municipio:

| Columna | Descripción |
|---------|-------------|
| `codigo_dane` | Código normalizado del municipio |
| `municipio` | Nombre del municipio |
| `departamento` | Departamento |
| `n_meses` | Meses incluidos en el histórico |
| `tasa_promedio` | Tasa histórica promedio |
| `p_25` | Percentil 25 |
| `p_50` | Percentil 50 |
| `p_75` | Percentil 75 |

Estos percentiles permiten clasificar el riesgo de cada municipio con una referencia local, en lugar de aplicar un único umbral nacional.

## Carril Municipio (Nivel Barrio)

Capa **independiente** del flujo nacional que agrega eventos a nivel de barrio
para los municipios configurados (fase 1: `76109` Buenaventura). Es el insumo
para disagregar la predicción municipal de XGBoost a barrios: **el modelo decide
cuántos hurtos habrá; los shares históricos deciden dónde**. No existe un
modelo ML barrial propio.

### Por qué no pasa por el bronce nacional

`BronzeEncoder` asigna códigos numéricos **alfabéticamente desde los `DISTINCT`
de cada fuente**. Si el feed de barrios se codificara por separado, `URBANA`
quedaría = 2 (en el nacional es = 3) y el `CASE` del silver leería mal los
porcentajes **en silencio**. Por eso el carril:

- **no** pasa por `BronzeEncoder`;
- valida el léxico del feed como **subconjunto** del nacional (`utils/domains.py: COLUMNS_LEXICO`);
- homologa **explícitamente** a los códigos alfabéticos verificados del bronce
  nacional, con `TRIM`/`UPPER` (atrapa el `' FEMENINO'` con espacio inicial del feed):

| Variable | Código → valor |
|---|---|
| `zona` | 1=OTRAS · 2=RURAL · 3=URBANA |
| `armas_medios` | 1=- · 2=ARMA BLANCA / CORTOPUNZANTE · 3=ARMA DE FUEGO · 4=CONTUNDENTES · 5=ESCOPOLAMINA · 6=LLAVE MAESTRA · 7=MEDIOS DE COMUNICACIÓN... · 8=NO REPORTADO · 9=SIN EMPLEO DE ARMAS |
| `genero` | 1=FEMENINO · 2=MASCULINO · 3=NO REPORTADO |
| `agrupa_edad` | 2=ADULTOS · 3=JOVENES · 4=MENORES |

### Tablas del carril

| Tabla | Rol |
|-------|-----|
| `{schema}.hurto_personas_municipio` | Fuente cruda del feed (eventos con `id_barrio`) |
| `{schema}.poblacion_municipio` | Población por barrio (cargada con `src/scripts/load_data.py`) |
| `{schema}.hurto_personas_bronce_municipio` | Salida: eventos homologados a la estructura del bronce nacional + llaves del carril |
| `{schema}.municipio_barrios_geo` | Geometrías de barrios (mapa/ficha) |

**Granularidad:** 1 fila = 1 evento (igual que `hurtos_personas_bronce`), con
las llaves del carril al final: `id_barrio` y `barrio`. El `id_barrio` es
`MD5(TRIM(UPPER(barrio)))` y **colisiona entre municipios** (hay barrios con
el mismo nombre en distintos municipios) — la llave efectiva es **compuesta**
(municipio + barrio).

### Creación de la tabla (SQL directo)

Fase 1 sin código mantenido: la tabla se crea con CTAS desde el editor de SQL
de Databricks. Reglas de homologación:

- **Columna inexistente en el feed** (`departamento`, `clase_sitio`, `dia`) → `NULL`.
- **Valor fuera del léxico** → `0` (0 no es código nacional; el `CASE` del silver
  lo ignora igual que ignora el `-` nacional).
- `tematica`: `'HURTO PERSONAS'` (feed) → `'HURTO A PERSONAS'` (nacional).
- Buckets de edad: menores `< 18` → 4 · jóvenes `18-28` → 3 · adultos `> 28` → 2.
- Si se agregan municipios al carril, basta un feed con `codigo_divipola`
  correcto y re-ejecutar el CTAS.

```sql
CREATE OR REPLACE TABLE workspace.territorial_inteligence.hurto_personas_bronce_municipio
USING DELTA
AS

SELECT
    CASE WHEN TRIM(UPPER(tematica)) = 'HURTO PERSONAS'
         THEN 'HURTO A PERSONAS' ELSE TRIM(UPPER(tematica)) END AS tematica,
    anio  AS anio,
    NULL AS departamento,
    codigo_divipola as codigo_dane,
    municipio AS municipio,
    CAST(NULL AS STRING) AS clase_sitio,
    CAST(NULL AS STRING) AS dia,
    CAST(cantidad AS BIGINT) AS cantidad,
    CASE SUBSTR(LOWER(TRIM(mes)), 1, 3)
        WHEN 'ene' THEN 1 WHEN 'feb' THEN 2 WHEN 'mar' THEN 3 WHEN 'abr' THEN 4
        WHEN 'may' THEN 5 WHEN 'jun' THEN 6 WHEN 'jul' THEN 7 WHEN 'ago' THEN 8
        WHEN 'sep' THEN 9 WHEN 'oct' THEN 10 WHEN 'nov' THEN 11 WHEN 'dic' THEN 12
    END AS mes,
    CAST(CASE TRIM(UPPER(genero))
        WHEN 'MASCULINO' THEN 2 WHEN 'FEMENINO' THEN 1 WHEN 'NO REPORTADO' THEN 3
        ELSE 0 END AS INT) AS genero,
    CAST(CASE TRIM(UPPER(zona))
        WHEN 'URBANA' THEN 3 WHEN 'RURAL' THEN 2 WHEN 'OTRAS' THEN 1
        ELSE 0 END AS INT) AS zona,
    CAST(CASE TRIM(UPPER(armas_medios))
        WHEN 'ARMA DE FUEGO' THEN 3
        WHEN 'ARMA BLANCA / CORTOPUNZANTE' THEN 2
        WHEN 'CONTUNDENTES' THEN 4
        WHEN 'ESCOPOLAMINA' THEN 5
        WHEN 'NO REPORTADO' THEN 8
        WHEN 'SIN EMPLEO DE ARMAS' THEN 9
        ELSE 0 END AS INT) AS armas_medios,
    CAST(CASE
        WHEN TRY_CAST(TRIM(edad) AS INT) IS NULL               THEN 0
        WHEN TRY_CAST(TRIM(edad) AS INT) < 18                  THEN 4
        WHEN TRY_CAST(TRIM(edad) AS INT) <= 28                 THEN 3
        ELSE 2
    END AS INT) AS agrupa_edad,
    TRIM(id_barrio) AS id_barrio,
    TRIM(UPPER(barrio)) AS barrio
FROM workspace.territorial_inteligence.hurto_personas_municipio
WHERE id_barrio IS NOT NULL AND TRIM(id_barrio) <> ''
```

### Validaciones post-creación

```sql
-- (a) Estructura: mismas columnas del bronce nacional + llaves del carril
DESCRIBE workspace.territorial_inteligence.hurto_personas_bronce_municipio

-- (b) Valores caídos en 0 (esperado 0 en las 4: el léxico del feed es subconjunto)
SELECT
    SUM(CASE WHEN genero = 0 THEN 1 ELSE 0 END)        AS genero_cero,
    SUM(CASE WHEN zona = 0 THEN 1 ELSE 0 END)          AS zona_cero,
    SUM(CASE WHEN armas_medios = 0 THEN 1 ELSE 0 END)  AS armas_cero,
    SUM(CASE WHEN agrupa_edad = 0 THEN 1 ELSE 0 END)   AS edad_cero
FROM workspace.territorial_inteligence.hurto_personas_bronce_municipio

-- (c) Conservación y consistencia
SELECT COUNT(*) AS filas, SUM(cantidad) AS eventos,
       COUNT(DISTINCT CONCAT(CAST(codigo_dane AS STRING), '-', id_barrio)) AS barrios,
       COUNT(DISTINCT tematica) AS tematicas,
       SUM(CASE WHEN mes IS NULL THEN 1 ELSE 0 END) AS mes_null
FROM workspace.territorial_inteligence.hurto_personas_bronce_municipio

-- (d) Brecha sin barrio en el feed crudo (transparencia)
SELECT CAST(codigo_divipola AS STRING) AS codigo_dane, COUNT(*) AS sin_barrio
FROM workspace.territorial_inteligence.hurto_personas_municipio
WHERE id_barrio IS NULL OR TRIM(id_barrio) = ''
GROUP BY CAST(codigo_divipola AS STRING)
```

| Query | Esperado (fase 1, Buenaventura) |
|-------|--------------------------------|
| (a) | `tematica, anio, departamento, codigo_dane, municipio, clase_sitio, dia, cantidad, mes, genero, zona, armas_medios, agrupa_edad, id_barrio, barrio` |
| (b) | **0 × 4** (si da >0, hay una etiqueta nueva en el feed) |
| (c) | filas = eventos = **3,531** · barrios 203 · tematicas 1 · mes_null 0 |
| (d) | ~1,438 eventos sin barrio (quedan fuera del carril) |

### Limitaciones de datos

- El feed `hurto_personas_municipio` cubre **2020-2024** (sin 2025-2026);
  cualquier predicción disagregada arrastra ese desfase, que se expone en la
  salida (`meses_desfase`).
- Solo el 35.8% de los eventos cruzan con `poblacion_municipio` (barrios con
  población conocida); los micro-barrios sin población quedan sin tasa.



## Ejecución del Pipeline

### Ejecución completa

```bash
uv run python -m territorial_inteligence_methodology.main --step all
```

### Ejecución por pasos

```bash
# 1. Validar calidad de datos
uv run python -m territorial_inteligence_methodology.main --step quality

# 2. Codificar variables categóricas
uv run python -m territorial_inteligence_methodology.main --step encode

# 3. Generar tabla silver (features temporales)
uv run python -m territorial_inteligence_methodology.main --step silver

# 4. Entrenar modelo y predecir 2025
uv run python -m territorial_inteligence_methodology.main --step model

# 5. Consolidar tablas gold, tasas, errores y niveles de riesgo
uv run python -m territorial_inteligence_methodology.main --step consolidated
```

Para cargar los festivos nacionales de Colombia de 2025 y 2026:

```bash
uv run python src/scripts/load_holidays_data.py
```

### Parámetros adicionales

```bash
# Sobreescribir catálogo y schema
uv run python -m territorial_inteligence_methodology.main --step all \
  --catalog registros_criminales \
  --schema mi_usuario
```

## API de consulta (Flask sobre el SQL warehouse de Databricks)

### Ejecución en Databricks

```bash
# Deploy al workspace (desarrollo)
databricks bundle deploy --target dev

# Ejecutar el job (todas las tasks: fase A + fase B)
databricks bundle run territorial_inteligence_pipeline -t dev

# Deploy a producción
databricks bundle deploy --target prod
```

## API de consulta (Flask sobre el SQL warehouse de Databricks)

API REST en `src/territorial_inteligence_methodology/app/` que expone las
salidas gold de las dos fases y las geometrías. Cada endpoint ejecuta SQL
contra el warehouse (`workspace.territorial_inteligence`) vía `SqlExecutor`.
Detalle completo en `tecnical-analisis.md`, §8.6.

```bash
# Arranque (producción: gunicorn · dev: API_DEBUG=true)
uv run --frozen python src/scripts/run_api.py
# o con el wheel instalado:
uv run --frozen api
```

Base: `http://localhost:3300/api/territorial_intelligence` — puerto `$PORT` (default 3300).

| Ruta | Devuelve |
|---|---|
| `/gold`, `/gold/<mes>` | consolidado histórico nacional pred/real (fase A) |
| `/prediction` | predicción del último mes (fase A) |
| `/prediction/neighborhood[ /current /<mes>]` | gold barrial Buenaventura 2020-2024 + forecast (fase B) |
| `/department` · `/neighborhood` · `/department/<cod>/municipality` · `/municipality/<codigo_dane>` | GeoJSON de territorios |

Los 10 endpoints fueron smoke-testeados contra el workspace el 2026-10-28 (200 OK).

### Despliegue sobre Databricks (Databricks App)

La API corre como **Databricks App** en el workspace
(https://territorial-api-7474655446021466.aws.databricksapps.com). Para
actualizarla tras cambiar código:

```bash
bash src/scripts/build_api_app.sh
rm -rf /tmp/api-src && mkdir -p /tmp/api-src && rsync -a apps/territorial-api/ /tmp/api-src/
databricks workspace import-dir /tmp/api-src \
  /Workspace/Users/juan.ulloa@usantotomas.edu.co/territorial-api-src \
  --overwrite --profile DEFAULT
databricks apps deploy territorial-api --auto-approve --mode SNAPSHOT \
  --source-code-path /Workspace/Users/juan.ulloa@usantotomas.edu.co/territorial-api-src \
  --profile DEFAULT
```

Ver detalle y pendientes en `tecnical-analisis.md`, §8.6.

### Frontend público (Angular/Vercel)

Existe un **proxy serverless** en `api-public/` (Vercel Function) que consulta
el SQL warehouse con un service principal M2M — sin login de Databricks. Copiar
`api-public/vercel/api/dbx.mjs` + `vercel.json` al proyecto Angular y definir
los env de Vercel. Instrucciones: `api-public/README.md`.

## Datos

### Diccionario de variables

El detalle completo de las variables que genera Silver y consume el modelo
(bloques: autorregresivo, cobertura, momentum, composición, estructurales y
target) está en **`diccionario_datos.md`**.

### Fuente

- **Hurtos contra personas:** https://archive.ics.uci.edu/ (dataset de hurtos en Colombia)
- **Consolidado municipal:** Variables socioeconómicas por municipio (población, PDET, ZOMAC, riesgo, PIB, geolocalización)

### Tablas en Unity Catalog

| Tabla | Descripción |
|-------|-------------|
| `{schema}.hurtos_personas_bronce` | Datos crudos codificados y normalizados |
| `{schema}.hurtos_personas_silver` | Datos agregados por municipio/mes con features temporales |
| `{schema}.consolidado` | Variables socioeconómicas y geográficas por municipio |
| `{schema}.hurto_personas_prediccion` | Predicciones de tasa de hurtos generadas por XGBoost |
| `{schema}.thresholds` | Percentiles históricos por municipio |
| `{schema}.hurtos_personas_consolidado` | Tasas reales y predichas, errores y métricas de evaluación |
| `{schema}.hurtos_personas_gold` | Tabla final para análisis y dashboard |
| `{schema}.municipios_geo` | Geometrías de municipios para mapas coropléticos |
| `{schema}.departamentos_geo` | Geometrías de departamentos para mapas coropléticos |
| `{schema}.holidays` | Festivos nacionales de Colombia por fecha |
| `{schema}.hurto_personas_municipio` | Feed crudo de hurtos a nivel barrial (fuente del carril municipio) |
| `{schema}.hurto_personas_bronce_municipio` | Carril municipio: barrio × mes con composición del hecho |
| `{schema}.poblacion_municipio` | Población por barrio (cargada con `load_data.py`) |
| `{schema}.municipio_barrios_geo` | Geometrías de barrios por municipio |

### Festividades

La tabla `holidays` contiene los festivos nacionales de Colombia. Puede relacionarse con las tablas mensuales usando:

```sql
ON gold.anio = YEAR(holidays.date)
AND gold.mes = MONTH(holidays.date)
```

Esto permite identificar meses con festivos, comparar tasas entre meses con y sin festivos y mostrar los nombres de las festividades en el dashboard.

## Dashboard AI/BI

El proyecto incluye un dashboard de Databricks con dos hojas recomendadas:

### Vista Gerencial

- KPIs de total de hurtos, tasa promedio, MAE y municipios de alto riesgo.
- Tendencia mensual de tasa real frente a tasa predicha.
- Distribución de riesgo por departamento.
- Top de departamentos por tasa.
- Mapa coroplético departamental.
- Comparativa de tasa entre meses con y sin festividades.

### Detalle Municipal

- Filtros por departamento, año, mes y nivel de riesgo.
- Tabla detallada por municipio y periodo.
- Mapa coroplético municipal.
- Tendencia mensual real frente a predicha.
- Comparativa departamental frente al promedio nacional.
- Análisis de meses con festividades.

Los mapas utilizan las tablas `municipios_geo` y `departamentos_geo`, que almacenan geometrías en EPSG:4326.

## Dependencias Principales

| Paquete | Versión | Uso |
|---------|---------|-----|
| `xgboost` | ≥3.4.1 | Modelo de regresión |
| `scikit-learn` | ≥1.6.1 | Métricas y preprocessing |
| `joblib` | ≥1.4.2 | Persistencia del modelo |
| `pandas` | ≥2.2.3 | Manipulación de datos |
| `databricks-sdk` | ≥0.133.0 | Conexión con Databricks |
| `pyarrow` | ≥19.0.1 | Lectura de Parquet |
| `python-dotenv` | ≥1.0.0 | Variables de entorno |

## Testing

```bash
# Ejecutar tests (requiere Databricks Connect configurado)
uv run pytest
```

## Notas de Desarrollo

- **Fuga de datos:** Las features temporales (media móvil, tendencia, crecimiento) incluyen el mes actual. Para un modelo más limpio, considerar recalcular con `ROWS BETWEEN N PRECEDING AND 1 PRECEDING`.
- **Modelo guardado:** El archivo `xgboost_hurtos.pkl` se genera automáticamente al ejecutar `--step model`.
- **Tabla de predicciones:** Se crea/reescribe con `CREATE OR REPLACE TABLE`, por lo que siempre refleja la última ejecución.
- **Códigos municipales:** Deben normalizarse a cinco dígitos antes de realizar JOINs con población, umbrales o geometrías.
- **Geometrías:** Las columnas `GEOMETRY` no deben utilizarse directamente en `GROUP BY`; para mapas agregados se recomienda usar una metric view o agregar la geometría con `FIRST()`/`ST_UNION_AGG()`.
