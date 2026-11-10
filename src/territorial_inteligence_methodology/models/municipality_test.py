import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor
from territorial_inteligence_methodology.models.xgboots_model import XGBoostModel


class MunicipalityModel:

    VARS_ID = ['municipio', 'codigo_dane', 'mes', 'anio']

    OUTPUT_COLS = [
        'codigo_dane',
        'id_barrio',
        'barrio',
        'anio',
        'mes',
        'pob_dane_2025',
        'conteo_predicho',
        'tasa_predicha',
    ]

    PRED_TABLE = 'hurto_personas_prediccion_municipio'
    SILVER_TABLE = 'hurto_personas_silver_municipio'
    THRESHOLD_TABLE = 'thresholds_shrinkage_municipio'

    GATE_SPEARMAN_MIN = 0.35
    GATE_MAE_REL_MAX = 0.25

    LAMBDA_MAX_CERO_PLAUSIBLE = 3.0

    def __init__(self, random_state=42):
        self.config = Config()
        self.sql_executor = SqlExecutor()
        self.catalog = self.config.CATALOG
        self.schema = self.config.SCHEMA
        self.table_silver = self.SILVER_TABLE
        self.table_pred = self.PRED_TABLE
        self.table_thresholds = self.THRESHOLD_TABLE
        self.model_path = './src/territorial_inteligence_methodology/models/xgboost_hurtos.pkl'
        self.random_state = random_state
        self.mes_actual = datetime.now().month
        self.anio_actual = datetime.now().year
        self.base = None

    def municipios_neighborhood(self):
        table = f"{self.catalog}.{self.schema}.{self.table_silver}"
        result = self.sql_executor.execute_sql_query(
            self.catalog,
            self.schema,
            f"SELECT DISTINCT CAST(codigo_dane AS STRING) AS codigo_dane FROM {table}"
        )
        return sorted(r['codigo_dane'] for r in result['data'])

    def generate_training_municipio(self):
        
        print("=" * 50)
        print("MUNICIPALITY MODEL - TRAINING BASE (SQL)")
        print("=" * 50)

        silver = f"{self.catalog}.{self.schema}.{self.table_silver}"
        training = f"{self.catalog}.{self.schema}.hurto_personas_training_municipio"
        poblacion = f"{self.catalog}.{self.schema}.poblacion_municipio"
        consolidado = f"{self.catalog}.{self.schema}.{self.config.CONSOLIDATED_TABLE_NAME}"

        query = f"""
            CREATE OR REPLACE TABLE {training}
            USING DELTA
            AS
            WITH munis AS (
                SELECT DISTINCT id_barrio, codigo_dane, anio, mes
                FROM {silver}
            ),
            ctx AS (
                SELECT * FROM {consolidado}
                WHERE CAST(codigo_municipio AS STRING) IN
                      (SELECT DISTINCT CAST(codigo_dane AS STRING) FROM {silver})
            ),
            hist AS (
                SELECT id_barrio,
                       SUM(CASE WHEN total_tematica IS NOT NULL THEN total_tematica ELSE 0 END) AS casos_hist,
                       COUNT(total_tematica) AS meses_ventana
                FROM {silver}
                WHERE anio BETWEEN 2020 AND 2024
                GROUP BY id_barrio
            )
            SELECT
                a.municipio,
                a.codigo_dane,
                NULL AS departamento,
                a.mes AS mes,
                a.anio AS anio,
                a.total_lag_1 AS total_lag_1,
                a.total_lag_2 AS total_lag_2,
                a.total_lag_3 AS total_lag_3,
                a.total_lag_6 AS total_lag_6,
                a.total_lag_12 AS total_lag_12,
                a.media_movil_3_meses AS media_movil_3_meses,
                a.media_movil_6_meses AS media_movil_6_meses,
                a.media_movil_12_meses AS media_movil_12_meses,
                a.n_obs_3 AS n_obs_3,
                a.n_obs_6 AS n_obs_6,
                a.n_obs_12 AS n_obs_12,
                a.historia_meses AS historia_meses,
                a.historia_insuficiente_3 AS historia_insuficiente_3,
                a.historia_insuficiente_6 AS historia_insuficiente_6,
                a.historia_insuficiente_12 AS historia_insuficiente_12,
                a.crecimiento_intermensual AS crecimiento_intermensual,
                a.crecimiento_interanual AS crecimiento_interanual,
                a.max_anio_anterior AS max_anio_anterior,
                a.max_mes_anterior AS max_mes_anterior,
                a.pct_zona_urbana_lag_1 AS pct_zona_urbana_lag_1,
                a.pct_zona_rural_lag_1 AS pct_zona_rural_lag_1,
                a.pct_arma_fuego_lag_1 AS pct_arma_fuego_lag_1,
                a.pct_arma_blanca_lag_1 AS pct_arma_blanca_lag_1,
                a.pct_arma_contundente_lag_1 AS pct_arma_contundente_lag_1,
                a.pct_genero_masculino_lag_1 AS pct_genero_masculino_lag_1,
                a.pct_genero_femenino_lag_1 AS pct_genero_femenino_lag_1,
                a.pct_edad_adultos_lag_1 AS pct_edad_adultos_lag_1,
                a.pct_edad_jovenes_lag_1 AS pct_edad_jovenes_lag_1,
                a.pct_edad_menores_lag_1 AS pct_edad_menores_lag_1,
                c.codigo_departamento,
                CAST(COALESCE(CAST(p.pob_2024 AS DOUBLE), CAST(p.pob_2023 AS DOUBLE), CAST(p.pob_2022 AS DOUBLE), CAST(p.pob_2021 AS DOUBLE), CAST(p.pob_2020 AS DOUBLE), 0) AS DOUBLE) AS pob_dane_2025,
                c.pdet,
                c.zomac,
                c.pdet_zomac,
                c.riesgo_victimizacion,
                c.pres_grup_armados,
                c.victimizacion_seguridad,
                c.pib2024_peso_relativo_mun_valor_agregado_departamental,
                c.art95_num1_base_datos_negativa_del_articulo_106_ley_1453_2011,
                c.longitud,
                c.latitud,
                c.cluster_rv,
                c.cluster_pga,
                c.cluster_vs,
                c.municipio_isla_area_no_municipalizada,
                LOG1P(CAST(COALESCE(CAST(p.pob_2024 AS DOUBLE), CAST(p.pob_2023 AS DOUBLE), CAST(p.pob_2022 AS DOUBLE), CAST(p.pob_2021 AS DOUBLE), CAST(p.pob_2020 AS DOUBLE), 0) AS DOUBLE)) AS log_pob_dane,
                (COALESCE(a.total_tematica, 0) / CAST(COALESCE(CAST(p.pob_2024 AS DOUBLE), CAST(p.pob_2023 AS DOUBLE), CAST(p.pob_2022 AS DOUBLE), CAST(p.pob_2021 AS DOUBLE), CAST(p.pob_2020 AS DOUBLE), 0) AS DOUBLE)) * 100000 AS target,
                a.id_barrio,
                a.barrio
            FROM {silver} a
            LEFT JOIN {poblacion} p
                ON p.id_barrio = a.id_barrio
            JOIN ctx c
                ON CAST(a.codigo_dane AS STRING) = CAST(c.codigo_municipio AS STRING)
            LEFT JOIN hist h ON h.id_barrio = a.id_barrio
            WHERE CAST(COALESCE(CAST(pob_2024 AS DOUBLE), CAST(pob_2023 AS DOUBLE), CAST(pob_2022 AS DOUBLE), CAST(pob_2021 AS DOUBLE), CAST(pob_2020 AS DOUBLE), 0) AS DOUBLE) > 0
              AND (
                      a.total_tematica IS NOT NULL
                   OR COALESCE(h.casos_hist, 0) / NULLIF(CAST(h.meses_ventana AS DOUBLE), 0)
                      <= 3.0
                  )
            """

        print(f"Generando {training} ...")
        self.sql_executor.execute_sql_query(self.catalog, self.schema, query)

        v = self.sql_executor.execute_sql_query(
            self.catalog, self.schema,
            f"""
                SELECT COUNT(*) AS filas,
                       COUNT(DISTINCT id_barrio) AS barrios,
                       MIN(anio) AS anio_min, MAX(anio) AS anio_max
                FROM {training}
            """
        )['data'][0]
        print(f"  Base generada: {v['filas']} filas · {v['barrios']} barrios · "
              f"anios {v['anio_min']}-{v['anio_max']}")
        return training

    def load_base(self):
        print("=" * 50)
        print("MUNICIPALITY MODEL - LOADING TRAINING BASE")
        print("=" * 50)

        table = f"{self.catalog}.{self.schema}.hurto_personas_training_municipio"
        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, f"SELECT * FROM {table}")
        self.base = pd.DataFrame(result['data'])

        numeric_prefixes = ('total_', 'media_', 'n_obs', 'historia', 'creci', 'max_', 'pct_')
        numeric_columns = self.base.columns[
            pd.Series(self.base.columns).str.startswith(numeric_prefixes)
        ].tolist() + ['mes', 'anio', 'target', 'pob_dane_2025']
        for column in numeric_columns:
            self.base[column] = pd.to_numeric(self.base[column], errors='coerce')

        print(f"  Base cargada: {len(self.base)} filas · "
              f"{self.base['id_barrio'].nunique()} barrios · "
              f"anios {int(self.base['anio'].min())}-{int(self.base['anio'].max())}")

        municipios = self.municipios_neighborhood()
        self.municipios = municipios
        print(f"  Municipios del carril (distinct en silver): {municipios}")
        return self.base

    def generate_shrinkage_thresholds(self):
        print("=" * 50)
        print("MUNICIPALITY MODEL - SHRINKAGE THRESHOLDS (BARRIO)")
        print("=" * 50)

        threshold = f"{self.catalog}.{self.schema}.{self.THRESHOLD_TABLE}"
        silver = f"{self.catalog}.{self.schema}.{self.table_silver}"
        pop = f"{self.catalog}.{self.schema}.poblacion_municipio"

        query = f"""
            CREATE OR REPLACE TABLE {threshold}
            USING DELTA
            AS
            WITH barrios AS (
                SELECT b.id_barrio,
                       MAX(b.barrio) AS barrio,
                       MAX(CAST(COALESCE(CAST(p.pob_2024 AS DOUBLE), CAST(p.pob_2023 AS DOUBLE), CAST(p.pob_2022 AS DOUBLE), CAST(p.pob_2021 AS DOUBLE), CAST(p.pob_2020 AS DOUBLE), 0) AS DOUBLE)) AS poblacion,
                       MAX(b.codigo_dane) AS codigo_dane
                FROM (SELECT id_barrio, MAX(barrio) AS barrio,
                             CAST(MAX(codigo_dane) AS STRING) AS codigo_dane
                      FROM {silver} GROUP BY id_barrio) b
                INNER JOIN {pop} p ON p.id_barrio = b.id_barrio
                WHERE CAST(COALESCE(CAST(p.pob_2024 AS DOUBLE), CAST(p.pob_2023 AS DOUBLE), CAST(p.pob_2022 AS DOUBLE), CAST(p.pob_2021 AS DOUBLE), CAST(p.pob_2020 AS DOUBLE), 0) AS DOUBLE) > 0
                GROUP BY b.id_barrio
            ),
            con_lambda AS (
                SELECT s.id_barrio,
                       COALESCE(s.total_tematica, 0) AS casos,
                       CASE WHEN s.total_tematica IS NOT NULL THEN 1 ELSE 0 END AS tiene_datos,
                       SUM(COALESCE(s.total_tematica, 0)) OVER (PARTITION BY s.id_barrio) AS casos_hist,
                       COUNT(*) OVER (PARTITION BY s.id_barrio) AS meses_ventana
                FROM {silver} s
            ),
            barrio_stats AS (
                SELECT
                    c.id_barrio,
                    MAX(b.barrio) AS barrio,
                    MAX(b.codigo_dane) AS codigo_dane,
                    MAX(b.poblacion) AS poblacion,
                    COUNT(*) AS n_meses,
                    CAST(SUM(c.casos) AS BIGINT) AS casos_totales,
                    CAST(MAX(b.poblacion) * COUNT(*) AS BIGINT) AS exposicion
                FROM con_lambda c
                INNER JOIN barrios b ON b.id_barrio = c.id_barrio
                WHERE c.tiene_datos = 1
                   OR (c.casos_hist / NULLIF(c.meses_ventana, 0))
                      <= {self.LAMBDA_MAX_CERO_PLAUSIBLE}
                GROUP BY c.id_barrio
            ),
            municipio_prior AS (
                SELECT
                    SUM(casos_totales) / SUM(exposicion) AS mu_m,
                    AVG(exposicion) AS avg_exposicion,
                    GREATEST(0,
                        VAR_POP(casos_totales / exposicion) -
                        (SUM(casos_totales) / SUM(exposicion)) / AVG(exposicion)
                    ) AS var_lambda_m
                FROM barrio_stats
            ),
            municipio_gamma AS (
                SELECT
                    mu_m,
                    CASE
                        WHEN var_lambda_m > 0 AND mu_m > 0 THEN mu_m * mu_m / var_lambda_m
                        ELSE mu_m * avg_exposicion
                    END AS alpha_m,
                    CASE
                        WHEN var_lambda_m > 0 AND mu_m > 0 THEN mu_m / var_lambda_m
                        ELSE avg_exposicion
                    END AS beta_m
                FROM municipio_prior
            ),
            barrio_posterior AS (
                SELECT
                    m.id_barrio, m.codigo_dane, m.barrio, m.poblacion, m.n_meses,
                    m.casos_totales, m.exposicion,
                    (g.alpha_m + m.casos_totales) / (g.beta_m + m.exposicion) AS tasa_post_lambda,
                    SQRT(
                        ((g.alpha_m + m.casos_totales) / (g.beta_m + m.exposicion)) * m.poblacion
                        * (g.beta_m + m.exposicion + m.poblacion) / (g.beta_m + m.exposicion)
                    ) AS conteo_sd
                FROM barrio_stats m
                CROSS JOIN municipio_gamma g
            )
            SELECT
                bp.id_barrio,
                bp.codigo_dane,
                bp.barrio,
                bp.poblacion,
                bp.n_meses,
                bp.casos_totales,
                ROUND(GREATEST(CEIL(bp.tasa_post_lambda * bp.poblacion
                                    - 0.6745 * bp.conteo_sd - 0.5), 0)
                      / bp.poblacion * 100000, 2) AS p_25,
                ROUND(GREATEST(CEIL(bp.tasa_post_lambda * bp.poblacion - 0.5), 0)
                      / bp.poblacion * 100000, 2) AS p_50,
                ROUND(GREATEST(CEIL(bp.tasa_post_lambda * bp.poblacion
                                    + 0.6745 * bp.conteo_sd - 0.5),
                               CASE WHEN bp.tasa_post_lambda > 0 THEN 1 ELSE 0 END)
                      / bp.poblacion * 100000, 2) AS p_75,
                ROUND(bp.exposicion / (g.beta_m + bp.exposicion), 4) AS shrinkage_factor,
                ROUND(bp.tasa_post_lambda * 100000, 2) AS tasa_shrink_promedio,
                'bayes_shrinkage_municipio' AS threshold_type,
                CURRENT_TIMESTAMP() AS created_at
            FROM barrio_posterior bp
            CROSS JOIN municipio_gamma g
        """

        print(f"\nGenerando {threshold} ...")
        self.sql_executor.execute_sql_query(self.catalog, self.schema, query)

        result = self.sql_executor.execute_sql_query(
            self.catalog, self.schema,
            f"SELECT COUNT(*) n, ROUND(MIN(p_25), 2) p25_min, "
            f"ROUND(MAX(p_75), 2) p75_max FROM {threshold}")
        print(f"  Thresholds (barrio) generados: {result['data']}")
        return threshold

    def score_data(self):
       
        if self.base is None:
            self.load_base()

        print("=" * 50)
        print("MUNICIPALITY MODEL - SCORING WITH XGBOOTS")
        print("=" * 50)

        model = self._load_model()
        X = XGBoostModel(random_state=self.random_state).prepare_X(self.base)
        self.base['tasa_predicha'] = model.predict(X)
        self.base['conteo_predicho'] = (
            self.base['tasa_predicha'] * self.base['pob_dane_2025'] / 100000.0
        )
        self.base['real_count'] = self.base['target'] * self.base['pob_dane_2025'] / 100000.0

        print(f"  Scoring listo: mediana conteo {self.base['conteo_predicho'].median():.3f} · "
              f"min {self.base['conteo_predicho'].min():.3f} · "
              f"max {self.base['conteo_predicho'].max():.2f}")
        return self.base

    
    def validate_scores(self):
        filas = len(self.base)
        nulos = int(self.base['conteo_predicho'].isna().sum())
        negativos = int((self.base['conteo_predicho'] < 0).sum())
        if nulos:
            raise ValueError(f'{nulos} conteos nulos en el scoring')
        print(f"  Validación: {filas} filas · {negativos} conteos negativos "
              f"(salida cruda del modelo, sin ajustar)")
        return True

    
    def evaluate_folds(self, hasta_anio_exclusivo=None):
        paneo = self.base
        if hasta_anio_exclusivo is not None:
            print(f"  Folds de gates: anios < {hasta_anio_exclusivo}")
            paneo = self.base[self.base['anio'] < hasta_anio_exclusivo]

        suma_score = paneo.groupby(['anio', 'mes'])['conteo_predicho'].sum()
        suma_target = paneo.groupby(['anio', 'mes'])['real_count'].sum()
        pares = pd.DataFrame({'p': suma_score, 'r': suma_target}).dropna()
        mae_mensual_rel = float(
            (pares['p'] - pares['r']).abs().mean() / max(pares['r'].mean(), 1e-9)
        )

        folds = []
        for anio in sorted(paneo['anio'].unique()):
            sub = paneo[paneo['anio'] == anio]
            rho = sub['conteo_predicho'].rank().corr(sub['real_count'].rank())
            folds.append({'anio': int(anio), 'spearman': rho})

        if hasta_anio_exclusivo is not None:
            pub = self.base[self.base['anio'] >= hasta_anio_exclusivo]
            rho_pub = pub['conteo_predicho'].rank().corr(pub['real_count'].rank())
            folds.append({'anio': int(hasta_anio_exclusivo),
                          'spearman': rho_pub, 'solo_referencia': True})

        print("  Backtest rolling:")
        for f in folds:
            tag = ' (solo referencia, no gatea)' if f.get('solo_referencia') else ''
            print(f"    {f['anio']}: spearman={f['spearman']:.3f}{tag}")
        print(f"    mae_rel mensual (años de gate) = {mae_mensual_rel:.3f}")
        return folds, mae_mensual_rel

    def check_gates(self, folds, mae_mensual_rel):
        fold_gates = [f for f in folds if not f.get('solo_referencia')]
        spearman_mediana = float(np.median([f['spearman'] for f in fold_gates]))

        fallidos = []
        if spearman_mediana < self.GATE_SPEARMAN_MIN:
            fallidos.append(f'spearman {spearman_mediana:.3f} < {self.GATE_SPEARMAN_MIN}')
        if mae_mensual_rel > self.GATE_MAE_REL_MAX:
            fallidos.append(f'mae_mensual_rel {mae_mensual_rel:.3f} > {self.GATE_MAE_REL_MAX}')

        if fallidos:
            raise ValueError('GATES REPROBADOS - NO SE PUBLICA: ' + '; '.join(fallidos))

        print(f"  Gates PASS: spearman mediana={spearman_mediana:.3f} · "
              f"mae_mensual_rel={mae_mensual_rel:.3f}")
        return {'spearman_mediana': spearman_mediana, 'mae_mensual_rel': mae_mensual_rel}

    
    def generate_gold_municipio(self):
       
        print("=" * 50)
        print("MUNICIPALITY MODEL - GOLD")
        print("=" * 50)

        pred = f"{self.catalog}.{self.schema}.{self.table_pred}"
        poblacion = f"{self.catalog}.{self.schema}.poblacion_municipio"
        threshold = f"{self.catalog}.{self.schema}.{self.table_thresholds}"
        gold_table = 'hurto_personas_gold_municipio'
        gold = f"{self.catalog}.{self.schema}.{gold_table}"
        consolidado = f"{self.catalog}.{self.schema}.consolidado"

        query = f"""
           CREATE OR REPLACE TABLE {gold}
            USING DELTA
            AS
            WITH calculo AS (
                SELECT
                    p.codigo_dane,
                    p.id_barrio,
                    p.barrio,
                    p.anio,
                    p.mes,
                    p.pob_dane_2025,
                    p.conteo_predicho,
                    p.tasa_predicha,
                    b.municipio,
                    s.tematica_mes_anterior,
                    t.p_25,
                    t.p_50,
                    t.p_75,
                    t.shrinkage_factor,
                    t.tasa_shrink_promedio,
                    t.threshold_type,
                       CASE WHEN s.anio IS NULL THEN NULL
                    WHEN s.total_tematica IS NOT NULL
                    THEN (s.total_tematica / p.pob_dane_2025) * 100000
                    ELSE 0.0
                    END AS tasa_real,
                    b.pres_grup_armados,
                    b.pdet_zomac,
                    b.riesgo_victimizacion,
                    b.municipio_isla_area_no_municipalizada
                FROM {pred} p
                INNER JOIN {threshold} t
                    ON t.id_barrio = p.id_barrio
                LEFT JOIN {self.catalog}.{self.schema}.{self.table_silver} s
                    ON s.id_barrio = p.id_barrio
                   AND s.anio = p.anio AND s.mes = p.mes
                LEFT JOIN {consolidado} b
                    ON p.codigo_dane = b.codigo_municipio
            )
            SELECT
                c.municipio,
                c.codigo_dane,
                c.id_barrio,
                c.barrio,
                c.anio,
                c.mes,
                c.pob_dane_2025,
                c.conteo_predicho,
                c.tematica_mes_anterior,
                c.p_25,
                c.p_50,
                c.p_75,
                c.shrinkage_factor,
                c.tasa_shrink_promedio,
                c.threshold_type,
                c.tasa_predicha,
                c.tasa_real,
                c.tasa_predicha - c.tasa_real           AS error,
                ABS(c.tasa_predicha - c.tasa_real)      AS abs_error,
                POWER(c.tasa_predicha - c.tasa_real, 2) AS sq_error,
                CASE
                    WHEN c.tasa_predicha >= c.p_75 THEN 'ALTO'
                    WHEN c.tasa_predicha >= c.p_50 THEN 'MEDIO'
                    ELSE 'BAJO'
                END AS nivel_riesgo_predicho,
                CASE
                    WHEN c.tasa_real IS NULL THEN NULL
                    WHEN c.tasa_real >= c.p_75 THEN 'ALTO'
                    WHEN c.tasa_real >= c.p_50 THEN 'MEDIO'
                    ELSE 'BAJO'
                END AS nivel_riesgo_real,
                c.pres_grup_armados,
                c.pdet_zomac,
                c.riesgo_victimizacion,
                c.municipio_isla_area_no_municipalizada
            FROM calculo c
            ORDER BY c.codigo_dane, c.id_barrio, c.anio, c.mes
        """

        print(f"Generando {gold} ...")
        self.sql_executor.execute_sql_query(self.catalog, self.schema, query)

        v = self.sql_executor.execute_sql_query(
            self.catalog, self.schema,
            f"""
                SELECT COUNT(*) AS filas,
                       COUNT(DISTINCT id_barrio) AS barrios,
                       SUM(CASE WHEN tasa_real IS NULL THEN 1 ELSE 0 END) AS sin_tasa,
                       SUM(CASE WHEN nivel_riesgo_real IS NULL THEN 1 ELSE 0 END) AS sin_nivel
                FROM {gold}
            """
        )['data'][0]
        d = self.sql_executor.execute_sql_query(
            self.catalog, self.schema,
            f"SELECT nivel_riesgo_real, COUNT(*) AS n FROM {gold} GROUP BY nivel_riesgo_real"
        )['data']
        print(f"  Validación: {v['filas']} filas · {v['barrios']} barrios · "
              f"sin tasa (micro-barrio): {v['sin_tasa']} · sin nivel: {v['sin_nivel']}")
        print(f"  Distribución nivel_riesgo: {d}")
        return gold

    def score_forecast(self):
        print("=" * 50)
        print("MUNICIPALITY MODEL - FORECAST (MES ACTUAL)")
        print(f"  mes = {self.mes_actual} · anio = {self.anio_actual}")
        print("=" * 50)

        silver = f"{self.catalog}.{self.schema}.{self.table_silver}"
        poblacion = f"{self.catalog}.{self.schema}.poblacion_municipio"
        consolidado = f"{self.catalog}.{self.schema}.{self.config.CONSOLIDATED_TABLE_NAME}"

        query = f"""
            WITH ventanas AS (
                SELECT
                    h.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY h.codigo_dane, h.id_barrio
                        ORDER BY h.anio DESC, h.mes DESC
                    ) AS rn,
                    LAG(COALESCE(h.total_tematica, 0), 5) OVER w AS lag_5,
                    LAG(COALESCE(h.total_tematica, 0), 11) OVER w AS lag_11,
                    AVG(COALESCE(h.total_tematica, 0)) OVER (
                        PARTITION BY codigo_dane, id_barrio ORDER BY anio, mes
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS media_3_fwd,
                    AVG(COALESCE(h.total_tematica, 0)) OVER (
                        PARTITION BY codigo_dane, id_barrio ORDER BY anio, mes
                        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS media_6_fwd,
                    AVG(COALESCE(h.total_tematica, 0)) OVER (
                        PARTITION BY codigo_dane, id_barrio ORDER BY anio, mes
                        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS media_12_fwd,
                    COUNT(h.total_tematica) OVER (
                        PARTITION BY codigo_dane, id_barrio ORDER BY anio, mes
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS n_obs_3_fwd,
                    COUNT(h.total_tematica) OVER (
                        PARTITION BY codigo_dane, id_barrio ORDER BY anio, mes
                        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS n_obs_6_fwd,
                    COUNT(h.total_tematica) OVER (
                        PARTITION BY codigo_dane, id_barrio ORDER BY anio, mes
                        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS n_obs_12_fwd,
                    COUNT(h.total_tematica) OVER (
                        PARTITION BY codigo_dane, id_barrio) AS historia_fwd
                FROM {silver} h
                WINDOW w AS (PARTITION BY codigo_dane, id_barrio ORDER BY anio, mes)
            ),
            ultimo_mes AS (
                SELECT * FROM ventanas WHERE rn = 1
            ),
            pop AS (
                SELECT id_barrio, MAX(CAST(COALESCE(CAST(pob_2024 AS DOUBLE), CAST(pob_2023 AS DOUBLE), CAST(pob_2022 AS DOUBLE), CAST(pob_2021 AS DOUBLE), CAST(pob_2020 AS DOUBLE), 0) AS DOUBLE)) AS poblacion
                FROM {poblacion} GROUP BY id_barrio
            ),
            ctx AS (
                SELECT * FROM {consolidado}
                WHERE CAST(codigo_municipio AS STRING) IN
                      (SELECT DISTINCT CAST(codigo_dane AS STRING) FROM {silver})
            )
            SELECT
                u.municipio,
                u.codigo_dane,
                NULL AS departamento,
                {self.mes_actual} AS mes,
                {self.anio_actual} AS anio,
                COALESCE(u.total_tematica, 0) AS total_lag_1,
                u.total_lag_1 AS total_lag_2,
                u.total_lag_2 AS total_lag_3,
                u.lag_5 AS total_lag_6,
                u.lag_11 AS total_lag_12,
                u.media_3_fwd AS media_movil_3_meses,
                u.media_6_fwd AS media_movil_6_meses,
                u.media_12_fwd AS media_movil_12_meses,
                -- FIX fuga (A3) + desfasaje (A2): mismas fórmulas que el train, evaluadas en t=M
                CASE
                    WHEN u.total_lag_1 > 0
                    THEN (u.total_tematica / u.total_lag_1 - 1) * 100
                    ELSE NULL
                END AS crecimiento_intermensual,
                CASE
                    WHEN u.lag_11 > 0
                    THEN (u.total_tematica / u.lag_11 - 1) * 100
                    ELSE NULL
                END AS crecimiento_interanual,
                CAST(u.total_tematica - u.total_lag_1 AS INT) AS max_mes_anterior,
                CAST(u.total_tematica - u.lag_11 AS INT) AS max_anio_anterior,
                u.n_obs_3_fwd AS n_obs_3,
                u.n_obs_6_fwd AS n_obs_6,
                u.n_obs_12_fwd AS n_obs_12,
                u.historia_fwd AS historia_meses,
                IF(u.n_obs_3_fwd < 3, 1, 0) AS historia_insuficiente_3,
                IF(u.n_obs_6_fwd < 6, 1, 0) AS historia_insuficiente_6,
                IF(u.n_obs_12_fwd < 12, 1, 0) AS historia_insuficiente_12,
                u.pct_zona_urbana AS pct_zona_urbana_lag_1,
                u.pct_zona_rural AS pct_zona_rural_lag_1,
                u.pct_arma_fuego AS pct_arma_fuego_lag_1,
                u.pct_arma_blanca AS pct_arma_blanca_lag_1,
                u.pct_arma_contundente AS pct_arma_contundente_lag_1,
                u.pct_genero_masculino AS pct_genero_masculino_lag_1,
                u.pct_genero_femenino AS pct_genero_femenino_lag_1,
                u.pct_edad_adultos AS pct_edad_adultos_lag_1,
                u.pct_edad_jovenes AS pct_edad_jovenes_lag_1,
                u.pct_edad_menores AS pct_edad_menores_lag_1,
                c.codigo_departamento,
                CAST(COALESCE(CAST(p.pob_2024 AS DOUBLE), CAST(p.pob_2023 AS DOUBLE), CAST(p.pob_2022 AS DOUBLE), CAST(p.pob_2021 AS DOUBLE), CAST(p.pob_2020 AS DOUBLE), 0) AS DOUBLE) AS pob_dane_2025,
                c.pdet,
                c.zomac,
                c.pdet_zomac,
                c.riesgo_victimizacion,
                c.pres_grup_armados,
                c.victimizacion_seguridad,
                c.pib2024_peso_relativo_mun_valor_agregado_departamental,
                c.art95_num1_base_datos_negativa_del_articulo_106_ley_1453_2011,
                c.longitud,
                c.latitud,
                c.cluster_rv,
                c.cluster_pga,
                c.cluster_vs,
                c.municipio_isla_area_no_municipalizada,
                LOG1P(CAST(COALESCE(CAST(p.pob_2024 AS DOUBLE), CAST(p.pob_2023 AS DOUBLE), CAST(p.pob_2022 AS DOUBLE), CAST(p.pob_2021 AS DOUBLE), CAST(p.pob_2020 AS DOUBLE), 0) AS DOUBLE)) AS log_pob_dane,
                u.id_barrio,
                u.barrio
            FROM ultimo_mes u
            LEFT JOIN pop p ON p.id_barrio = u.id_barrio
            LEFT JOIN ctx c
                ON CAST(u.codigo_dane AS STRING) = CAST(c.codigo_municipio AS STRING)
            WHERE CAST(COALESCE(CAST(pob_2024 AS DOUBLE), CAST(pob_2023 AS DOUBLE), CAST(pob_2022 AS DOUBLE), CAST(pob_2021 AS DOUBLE), CAST(pob_2020 AS DOUBLE), 0) AS DOUBLE) > 0
        """
        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        df_forecast = pd.DataFrame(result['data'])

        numeric_prefixes = ('total_', 'media_', 'n_obs', 'historia', 'creci', 'max_', 'pct_')
        numeric_columns = df_forecast.columns[
            pd.Series(df_forecast.columns).str.startswith(numeric_prefixes)
        ].tolist() + ['mes', 'anio', 'pob_dane_2025']
        for column in numeric_columns:
            df_forecast[column] = pd.to_numeric(df_forecast[column], errors='coerce')

        print(f"  Features del forecast: {len(df_forecast)} barrios")

        model = self._load_model()
        X = XGBoostModel(random_state=self.random_state).prepare_X(df_forecast)
        df_forecast['tasa_predicha'] = model.predict(X)
        df_forecast['conteo_predicho'] = (
            df_forecast['tasa_predicha'] * df_forecast['pob_dane_2025'] / 100000.0
        )
        df_forecast['estado'] = 'MES_EN_CURSO'

        print(f"  Forecast listo: mediana conteo {df_forecast['conteo_predicho'].median():.3f} · "
              f"max {df_forecast['conteo_predicho'].max():.2f}")
        self.forecast = df_forecast
        return df_forecast
    def _load_model(self):
        """Ruta de carga del pkl: volumen UC compartido entre tasks,
        fallback a archivo local (dev)."""
        from territorial_inteligence_methodology.utils.model_store import ModelStore
        return ModelStore().load()


    @staticmethod
    def _sql_literal(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "NULL"
            if isinstance(v, (int, float, np.integer, np.floating)):
                return str(v)
            return "'" + str(v).replace("'", "''") + "'"
    
    def write_predictions_table(self, df_pred):
            cols = list(df_pred.columns)
            rows = []
            for _, r in df_pred.iterrows():
                vals = ", ".join(self._sql_literal(r[c]) for c in cols)
                rows.append(f"({vals})")
            values = ",\n".join(rows)
            out_table = f"{self.catalog}.{self.schema}.{self.table_pred}"
            query = f"""
                CREATE OR REPLACE TABLE {out_table} USING DELTA AS
                SELECT * FROM VALUES\n{values}\nAS t({', '.join(cols)})
            """
            print(f"\nEscribiendo tabla: {out_table} ({len(df_pred)} filas)...")
            self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
            print("  Tabla creada .....")
            return out_table
    
        
        
    def write_forecast_table(self, df_forecast):
        out_table = f"{self.catalog}.{self.schema}.hurto_personas_forecast_municipio"
        cols = ['codigo_dane', 'id_barrio', 'barrio', 'anio', 'mes',
                'pob_dane_2025', 'conteo_predicho', 'tasa_predicha', 'estado']
        df_pred = df_forecast[cols].round({'conteo_predicho': 4, 'tasa_predicha': 4})

        col_names = list(df_pred.columns)
        rows = []
        for _, r in df_pred.iterrows():
            vals = ", ".join(self._sql_literal(r[c]) for c in col_names)
            rows.append(f"({vals})")
        values = ",\n".join(rows)
        query = f"""
            CREATE OR REPLACE TABLE {out_table} USING DELTA AS
            SELECT * FROM VALUES\n{values}\nAS t({', '.join(col_names)})
        """
        print(f"\nEscribiendo tabla: {out_table} ({len(df_pred)} filas)...")
        self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        print("  Tabla creada .....")
        return out_table

    
    def consolidar_predicciones(self):
        print("=" * 50)
        print("MUNICIPALITY MODEL - CONSOLIDATE PREDICTIONS")
        print("=" * 50)

        pred = f"{self.catalog}.{self.schema}.{self.table_pred}"
        forecast = f"{self.catalog}.{self.schema}.hurto_personas_forecast_municipio"

        query = f"""
            CREATE OR REPLACE TABLE {pred}
            USING DELTA
            AS
            SELECT
                codigo_dane, id_barrio, barrio, anio, mes,
                pob_dane_2025, conteo_predicho, tasa_predicha
            FROM {forecast}
            UNION ALL
            SELECT
                codigo_dane, id_barrio, barrio, anio, mes,
                pob_dane_2025, conteo_predicho, tasa_predicha
            FROM {pred}
        """

        print(f"Consolidando {forecast} + histórico → {pred} ...")
        self.sql_executor.execute_sql_query(self.catalog, self.schema, query)

        v = self.sql_executor.execute_sql_query(
            self.catalog, self.schema,
            f"""
                SELECT COUNT(*) AS filas,
                       COUNT(DISTINCT CONCAT(CAST(anio AS STRING), '-', CAST(mes AS STRING))) AS periodos,
                       SUM(CASE WHEN anio = {self.anio_actual} THEN 1 ELSE 0 END) AS forecast_filas
                FROM {pred}
            """
        )['data'][0]
        print(f"  Consolidada: {v['filas']} filas · "
              f"historico (2020-2024) + forecast {v['forecast_filas']} filas del mes actual")
        return pred

    def run(self):
        datetime_now = datetime.now()
        print("=" * 50)
        print(f"MUNICIPALITY MODEL - RUN {datetime_now:%Y-%m-%d %H:%M}")
        print("=" * 50)

        self.generate_training_municipio()
        self.load_base()
        self.generate_shrinkage_thresholds()
        self.score_data()
        self.validate_scores()

        anio_publicar = int(self.base['anio'].max())
        folds, mae_mensual_rel = self.evaluate_folds(
            hasta_anio_exclusivo=anio_publicar)
        try:
            gates = self.check_gates(folds, mae_mensual_rel)
        except ValueError as e:
            # Gates reprobados = resultado válido (no publica), NO un crash:
            # el job termina limpio y las tablas de fase B quedan como están.
            print("=" * 50)
            print(f"GATES REPROBADOS - NO SE PUBLICA EN ESTA CORRIDA")
            print(f"  ({e})")
            print(f"  Las tablas de fase B (prediccion/forecast/gold municipio)")
            print(f"  se conservan sin cambios.")
            print("=" * 50)
            return None

        df_pred = self.base[self.base['anio'] == anio_publicar][self.OUTPUT_COLS].round(
            {'conteo_predicho': 4, 'tasa_predicha': 4})
        self.write_predictions_table(df_pred)
        df_forecast = self.score_forecast()
        self.write_forecast_table(df_forecast)
        
        self.consolidar_predicciones()
        self.generate_gold_municipio()

        print("=" * 50)
        print("MUNICIPALITY MODEL COMPLETE")
        print("=" * 50)
        return gates


def model_municipality_table():
    MunicipalityModel().run()
