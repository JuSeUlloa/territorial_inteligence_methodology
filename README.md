# Territorial Intelligence Methodology

Pipeline de datos para análisis territorial de hurtos en Colombia. Implementa la arquitectura **medallion** (bronze → silver → gold) en Databricks, con un modelo predictivo XGBoost que estima la tasa de hurtos por cada 100,000 habitantes por municipio y mes.

## Descripción del Proyecto

El pipeline procesa datos de hurtos contra personas, enriquece la información con variables socioeconómicas y geográficas de los municipios colombianos, y genera predicciones de la tasa de hurtos para el año 2025.

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

**Ubicación:** `src/territorial_inteligence_methodology/models/xgboots_model.py`

Modelo XGBoost que predice la tasa de hurtos para 2025. Detalles en la sección [Modelo Predictivo](#modelo-predictivo).

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
│   │   ├── models/
│   │   │   └── xgboots_model.py             # Modelo XGBoost
│   │   └── utils/
│   │       ├── client_api.py                # Cliente de APIs externas
│   │       ├── databricks_client.py         # Cliente singleton de Databricks
│   │       ├── domains.py                   # Mapeo de columnas originales
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
CATALOG=registros_criminales
SCHEMA=tu_usuario
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
```

### Parámetros adicionales

```bash
# Sobreescribir catálogo y schema
uv run python -m territorial_inteligence_methodology.main --step all \
  --catalog registros_criminales \
  --schema mi_usuario
```

### Ejecución en Databricks

```bash
# Deploy al workspace (desarrollo)
databricks bundle deploy --target dev

# Ejecutar el job
databricks bundle run

# Deploy a producción
databricks bundle deploy --target prod
```

## Datos

### Fuente

- **Hurtos contra personas:** https://archive.ics.uci.edu/ (dataset de hurtos en Colombia)
- **Consolidado municipal:** Variables socioeconómicas por municipio (población, PDET, ZOMAC, riesgo, PIB, geolocalización)

### Tablas en Unity Catalog

| Tabla | Descripción |
|-------|-------------|
| `{schema}.hurtos_personas_silver` | Datos agregados por municipio/mes con features temporales |
| `{schema}.consolidado` | Variables socioeconómicas y geográficas por municipio |
| `{schema}.hurto_personas_prediccion` | Predicciones de tasa de hurtos 2025 (tabla gold) |

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
