from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor
from datetime import datetime
from scipy.stats import nbinom
import pandas as pd
import numpy as np


class Consolidated:
  
    def __init__(self):
        self.config =Config()
        self.sql_executor= SqlExecutor()
        self.catalog = self.config.CATALOG
        self.schema = self.config.SCHEMA
        self.use_thresholds_shrinkage = self.config.USE_THRESHOLDS_SHRINKAGE
        self.prediction_table = self.config.PREDICTION_TABLE_NAME
        self.consolidated_table = self.config.CONSOLIDATED_TABLE_NAME
        self.silver_table = self.config.TABLE_SILVER_NAME
        self.LAMBDA_MAX_CERO_PLAUSIBLE = 3.0
        self.rate_table = 'hurtos_personas_consolidado'
        self.gold_table = 'hurtos_personas_gold'
        self.anio = datetime.now().year 

    def generate_threshold_table(self):
        tabla_consolidado =f"{self.catalog}.{self.schema}.{self.consolidated_table}"
        silver_table =f"{self.catalog}.{self.schema}.{self.config.TABLE_SILVER_NAME}"
        threshold =f"{self.catalog}.{self.schema}.thresholds"
                 
        return f"""
            CREATE OR REPLACE TABLE {threshold}
            USING DELTA
            AS
            WITH calendario AS (
            SELECT DISTINCT anio, mes
            FROM {silver_table}
            WHERE anio BETWEEN 2021 AND {(self.anio-1)}
        ),
        municipios AS (
            SELECT codigo_municipio, MAX(pob_dane_2025) AS pob_dane_2025
            FROM {tabla_consolidado}
            WHERE pob_dane_2025 > 0
            GROUP BY codigo_municipio
        ),
        spine AS (
            SELECT m.codigo_municipio, c.anio, c.mes, m.pob_dane_2025
            FROM municipios m
            CROSS JOIN calendario c
        ),
        tasas_mensuales AS (
            SELECT
                    s.codigo_municipio AS codigo_dane,
                    (COALESCE(a.total_tematica, 0) / s.pob_dane_2025) * 100000 AS tasa_real
                FROM spine s
                LEFT JOIN {silver_table} a
                    ON a.codigo_dane = s.codigo_municipio
                   AND a.anio = s.anio
                   AND a.mes = s.mes
        ),
        nombres AS (
            SELECT codigo_dane,
            MAX(municipio) AS municipio,
            MAX(departamento) AS departamento
            FROM {silver_table}
            GROUP BY codigo_dane
        )
        SELECT
            t.codigo_dane,
            n.municipio,
            n.departamento,
            COUNT(*) AS n_meses,
            ROUND(AVG(t.tasa_real), 2) AS tasa_promedio,
            ROUND(percentile(t.tasa_real, 0.25), 2) AS p_25,
            ROUND(percentile(t.tasa_real, 0.50), 2) AS p_50,
            ROUND(percentile(t.tasa_real, 0.75), 2) AS p_75
        FROM tasas_mensuales t
        LEFT JOIN nombres n ON t.codigo_dane = n.codigo_dane
        GROUP BY t.codigo_dane, n.municipio, n.departamento
        ORDER BY tasa_promedio DESC
        """
    def generate_threshold_shrinkage_table(self):
        tabla_consolidado =f"{self.catalog}.{self.schema}.{self.consolidated_table}"
        silver_table =f"{self.catalog}.{self.schema}.{self.config.TABLE_SILVER_NAME}"
        threshold =f"{self.catalog}.{self.schema}.thresholds_shrinkage"
                 
        return f"""
        CREATE OR REPLACE TABLE {threshold}
        USING DELTA
        AS
        WITH municipios AS (
            SELECT codigo_municipio, departamento, MAX(pob_dane_2025) AS poblacion
            FROM {tabla_consolidado}
            WHERE pob_dane_2025 > 0
            GROUP BY codigo_municipio, departamento
        ),
        calendario AS (
            SELECT explode(sequence(MAKE_DATE(2021, 1, 1),
                                    MAKE_DATE({self.anio - 1}, 12, 1),
                                    INTERVAL 1 MONTH)) AS fecha
        ),
        spine AS (
                        SELECT m.codigo_municipio, m.departamento, m.poblacion,
                               YEAR(c.fecha) AS anio, MONTH(c.fecha) AS mes
                        FROM municipios m CROSS JOIN calendario c
                    ),
        meses AS (
            SELECT s.*,
                   COALESCE(a.total_tematica, 0) AS casos,
                   CASE WHEN a.total_tematica IS NOT NULL THEN 1 ELSE 0 END AS tiene_datos
            FROM spine s
            LEFT JOIN {silver_table} a
                ON a.codigo_dane = s.codigo_municipio
               AND a.anio = s.anio AND a.mes = s.mes
        ),
        con_lambda AS (
            SELECT m.*,
                   SUM(casos) OVER (PARTITION BY codigo_municipio) AS casos_hist,
                   COUNT(*) OVER (PARTITION BY codigo_municipio) AS meses_ventana
            FROM meses m
        ),
        municipio_stats AS (
            SELECT
                codigo_municipio,
                departamento,
                MAX(poblacion) AS poblacion,
                COUNT(*) AS n_meses,
                CAST(SUM(casos) AS BIGINT) AS casos_totales,
                CAST(MAX(poblacion) * COUNT(*) AS BIGINT) AS exposicion
            FROM con_lambda
            WHERE tiene_datos = 1
               OR (casos_hist / NULLIF(meses_ventana, 0)) <= {self.LAMBDA_MAX_CERO_PLAUSIBLE}
            GROUP BY codigo_municipio, departamento
        ),
        depto_params AS (
            SELECT
                departamento,
                SUM(casos_totales) / SUM(exposicion) AS mu_d,
                AVG(exposicion) AS avg_exposicion,
                GREATEST(0,
                    VAR_POP(casos_totales / exposicion) -
                    (SUM(casos_totales) / SUM(exposicion)) / AVG(exposicion)
                ) AS var_lambda_d
            FROM municipio_stats
            GROUP BY departamento
        ),
        depto_gamma AS (
            SELECT
                departamento,
                mu_d,
                var_lambda_d,
                CASE
                    WHEN var_lambda_d > 0 AND mu_d > 0
                    THEN mu_d * mu_d / var_lambda_d
                    ELSE mu_d * avg_exposicion
                END AS alpha_d,
                CASE
                    WHEN var_lambda_d > 0 AND mu_d > 0
                    THEN mu_d / var_lambda_d
                    ELSE avg_exposicion
                END AS beta_d
            FROM depto_params
        ),
        municipio_posterior AS (
            SELECT
                m.codigo_municipio,
                m.departamento,
                m.poblacion,
                m.n_meses,
                m.casos_totales,
                m.exposicion,
                g.alpha_d,
                g.beta_d,
                (g.alpha_d + m.casos_totales) AS alpha_post,
                (g.beta_d + m.exposicion) AS beta_post,
                ((g.alpha_d + m.casos_totales) / (g.beta_d + m.exposicion)) * m.poblacion AS conteo_esperado,
                SQRT(
                    ((g.alpha_d + m.casos_totales) / (g.beta_d + m.exposicion)) * m.poblacion
                    * (g.beta_d + m.exposicion + m.poblacion) / (g.beta_d + m.exposicion)
                ) AS conteo_sd
            FROM municipio_stats m
            INNER JOIN depto_gamma g ON m.departamento = g.departamento
        ),
        nombres AS (
            SELECT codigo_dane,
                   MAX(municipio) AS municipio,
                   MAX(departamento) AS departamento
            FROM {silver_table}
            GROUP BY codigo_dane
        )
        SELECT
            mp.codigo_municipio AS codigo_dane,
            n.municipio,
            n.departamento,
            mp.poblacion,
            mp.n_meses,
            mp.casos_totales,
            ROUND(GREATEST(CEIL(mp.conteo_esperado - 0.6745 * mp.conteo_sd - 0.5), 0) / mp.poblacion * 100000, 2) AS p_25,
            ROUND(GREATEST(CEIL(mp.conteo_esperado - 0.5), 0) / mp.poblacion * 100000, 2) AS p_50,
            ROUND(GREATEST(CEIL(mp.conteo_esperado + 0.6745 * mp.conteo_sd - 0.5), CASE WHEN mp.conteo_esperado > 0 THEN 1 ELSE 0 END) / mp.poblacion * 100000, 2) AS p_75,
            ROUND(mp.exposicion / (g.beta_d + mp.exposicion), 4) AS shrinkage_factor,
            ROUND(mp.conteo_esperado / mp.poblacion * 100000, 2) AS tasa_shrink_promedio,
            'bayes_shrinkage' AS threshold_type,
            CURRENT_TIMESTAMP() AS created_at
        FROM municipio_posterior mp
        INNER JOIN depto_gamma g ON mp.departamento = g.departamento
        LEFT JOIN nombres n ON mp.codigo_municipio = n.codigo_dane
        """

    def calculate_shrinkage_threshold(self):
        tabla_consolidado =f"{self.catalog}.{self.schema}.{self.consolidated_table}"
        silver_table =f"{self.catalog}.{self.schema}.{self.config.TABLE_SILVER_NAME}"
        query = f"""
               WITH municipios AS (
                SELECT codigo_municipio, departamento, MAX(pob_dane_2025) AS poblacion
                FROM {tabla_consolidado}
                WHERE pob_dane_2025 > 0
                GROUP BY codigo_municipio, departamento
            ),
            calendario AS (
                SELECT explode(sequence(MAKE_DATE(2021, 1, 1),
                                        MAKE_DATE({self.anio - 1}, 12, 1),
                                        INTERVAL 1 MONTH)) AS fecha
            ),
            spine AS (
                SELECT m.codigo_municipio, m.departamento, m.poblacion,
                       YEAR(c.fecha) AS anio, MONTH(c.fecha) AS mes
                FROM municipios m CROSS JOIN calendario c
            ),
            meses AS (
                SELECT s.*,
                       COALESCE(a.total_tematica, 0) AS casos,
                       CASE WHEN a.total_tematica IS NOT NULL THEN 1 ELSE 0 END AS tiene_datos
                FROM spine s
                LEFT JOIN {silver_table} a
                    ON a.codigo_dane = s.codigo_municipio
                   AND a.anio = s.anio AND a.mes = s.mes
            ),
            con_lambda AS (
                SELECT m.*,
                       SUM(casos) OVER (PARTITION BY codigo_municipio) AS casos_hist,
                       COUNT(*) OVER (PARTITION BY codigo_municipio) AS meses_ventana
                FROM meses m
            )
            SELECT
                codigo_municipio,
                departamento,
                CAST(MAX(poblacion) AS BIGINT) AS poblacion,
                CAST(COUNT(*) AS INT) AS n_meses,
                CAST(SUM(casos) AS BIGINT) AS casos_totales,
                CAST(MAX(poblacion) * COUNT(*) AS BIGINT) AS exposicion
            FROM con_lambda
            WHERE tiene_datos = 1
               OR (casos_hist / NULLIF(meses_ventana, 0)) <= {self.LAMBDA_MAX_CERO_PLAUSIBLE}
            GROUP BY codigo_municipio, departamento"""
    
        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)

        results = []
        df = pd.DataFrame(result['data'], columns=[
                'codigo_municipio', 'departamento', 'poblacion', 'n_meses', 
                'casos_totales', 'exposicion'
        ])

        numeric_cols = ['poblacion', 'n_meses', 'casos_totales', 'exposicion']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['casos_totales', 'exposicion', 'poblacion'])
        
        for depto, grp in df.groupby('departamento'):
            mu = grp['casos_totales'].sum() / grp['exposicion'].sum()
            tasas_obs = grp['casos_totales'] / grp['exposicion']
            var_obs = tasas_obs.var(ddof=0)
            avg_exposicion = grp['exposicion'].mean()
            var_lambda = max(0, var_obs - mu / avg_exposicion)
        
            if var_lambda > 0 and mu > 0:
                alpha_d = mu**2 / var_lambda
                beta_d  = mu / var_lambda
            else:
                alpha_d = mu * avg_exposicion
                beta_d = avg_exposicion
        
            for _, row in grp.iterrows():
                alpha_post = alpha_d + row['casos_totales']
                beta_post = beta_d + row['exposicion']
                pob = row['poblacion']

                
                p_nb = beta_post / (beta_post + pob)
                esperado = alpha_post * (pob / beta_post)
                p25 = nbinom.ppf(0.25, alpha_post, p_nb) / pob * 100000
                p50 = nbinom.ppf(0.50, alpha_post, p_nb) / pob * 100000
                p75 = max(nbinom.ppf(0.75, alpha_post, p_nb), 1 if esperado > 0 else 0) / pob * 100000
                shrinkage = row['exposicion'] / (beta_d + row['exposicion'])
            
            results.append({
                'codigo_municipio': row['codigo_municipio'],
                'departamento': depto,
                'poblacion': row['poblacion'],
                'n_meses': row['n_meses'],
                'casos_totales': row['casos_totales'],
                'exposicion': row['exposicion'],
                'p_25': round(p25, 2),
                'p_50': round(p50, 2),
                'p_75': round(p75, 2),
                'shrinkage_factor': round(shrinkage, 4),
                'tasa_shrink_promedio': round(alpha_post / beta_post * 100000, 2),
                'threshold_type': 'bayes_shrinkage'
            })
    
        return pd.DataFrame(results)   
        

    def write_shrinkage_thresholds(self, df_thresholds: pd.DataFrame):
    
        table_name = f"{self.catalog}.{self.schema}.thresholds_shrinkage"
    
        cols = list(df_thresholds.columns)
        rows = []
        for _, r in df_thresholds.iterrows():
            vals = ", ".join(self._sql_literal(r[c]) for c in cols)
            rows.append(f"({vals})")
        values = ",\n".join(rows)
    
        query = f"""
            CREATE OR REPLACE TABLE {table_name} USING DELTA AS
            SELECT * FROM VALUES\n{values}\nAS t({', '.join(cols)})
        """
        self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        print(f"  Tabla {table_name} creada con {len(df_thresholds)} municipios")

   
    def consolidate_rate_table(self):

        rate = f"{self.catalog}.{self.schema}.{self.rate_table}"
        prediction = f"{self.catalog}.{self.schema}.{self.prediction_table}"
        prediction_forecast = f"{self.catalog}.{self.schema}.hurto_personas_forecast"
        silver = f"{self.catalog}.{self.schema}.{self.silver_table}"
        consolidado = f"{self.catalog}.{self.schema}.{self.consolidated_table}"
        threshold =f"{self.catalog}.{self.schema}.thresholds_shrinkage"  if self.use_thresholds_shrinkage else f"{self.catalog}.{self.schema}.thresholds"
        return f"""
            CREATE OR REPLACE TABLE {rate}
            USING DELTA 
            AS
            WITH historico AS (
            SELECT codigo_dane,
                   SUM(total_tematica) AS casos_hist,
                   COUNT(*) AS meses_ventana
            FROM {silver}
            WHERE anio BETWEEN 2021 AND {self.anio - 1}
            GROUP BY codigo_dane
            ),
            real_{self.anio} AS (
                SELECT
                    a.codigo_dane,
                    a.mes,
                    a.anio,
                    COALESCE(a.total_tematica,
                    CASE WHEN COALESCE(h.casos_hist,0) / NULLIF(h.meses_ventana,0) <= 3.0
                         THEN 0 END) AS total_tematica,
                    (COALESCE(a.total_tematica,
                     CASE WHEN COALESCE(h.casos_hist,0) / NULLIF(h.meses_ventana,0) <= 3.0
                          THEN 0 END) / c.pob_dane_2025) * 100000 AS tasa_real,
                    a.tematica_mes_anterior,
                    c.pob_dane_2025
                FROM {silver} a
                INNER JOIN {consolidado} c
                    ON a.codigo_dane = c.codigo_municipio
                LEFT JOIN historico h ON a.codigo_dane = h.codigo_dane
                WHERE a.anio >= {self.anio} and c.pob_dane_2025 > 0
            ),
            predictions AS (
                SELECT municipio,codigo_dane,departamento,mes,anio,tasa_predicha
                FROM {prediction}
                UNION ALL
                SELECT municipio,codigo_dane,departamento,mes,anio,tasa_predicha
                FROM {prediction_forecast}
                )
            SELECT
                p.municipio,
                p.codigo_dane,
                p.departamento,
                CAST(p.mes as INT) as mes,
                CAST(p.anio as INT) as anio,
                p.tasa_predicha,
                r.tasa_real,
                r.total_tematica,
                r.tematica_mes_anterior,
                r.pob_dane_2025,
                (p.tasa_predicha - r.tasa_real)         AS error,
                ABS(p.tasa_predicha - r.tasa_real)      AS abs_error,
                POWER(p.tasa_predicha - r.tasa_real, 2) AS sq_error,
                CASE
                    WHEN r.tasa_real <> 0
                    THEN ABS(p.tasa_predicha - r.tasa_real) / ABS(r.tasa_real) * 100
                    ELSE NULL
                END                                     AS ape_pct,
                CASE
                    WHEN r.tasa_real IS NULL THEN 'SIN_DATO'
                    WHEN t.p_25 IS NULL THEN 'SIN_UMBRAL'
                    WHEN r.tasa_real <= t.p_25 THEN 'BAJO'
                    WHEN r.tasa_real < t.p_75 THEN 'MEDIO'
                    ELSE 'ALTO'
                END AS nivel_riesgo_real,
                CASE
                    WHEN p.tasa_predicha IS NULL THEN 'SIN_DATO'
                    WHEN t.p_25 IS NULL THEN 'SIN_UMBRAL'
                    WHEN p.tasa_predicha <= t.p_25 THEN 'BAJO'
                    WHEN p.tasa_predicha < t.p_75 THEN 'MEDIO'
                    ELSE 'ALTO'
                END AS nivel_riesgo_predicho,
                t.p_25,
                t.p_50,
                t.p_75
            FROM predictions p
            LEFT JOIN real_{self.anio} r
                ON CAST(p.codigo_dane AS STRING) = CAST(r.codigo_dane AS STRING)
                AND CAST(p.mes AS INT)  = CAST(r.mes AS INT)
                AND CAST(p.anio AS INT) = CAST(r.anio AS INT)
            LEFT JOIN {threshold} t 
            ON p.codigo_dane = t.codigo_dane

        """
    def generate_gold_table(self):

        rate = f"{self.catalog}.{self.schema}.{self.rate_table}"
        gold = f"{self.catalog}.{self.schema}.{self.gold_table}"
        consolidado = f"{self.catalog}.{self.schema}.{self.consolidated_table}"

        return f"""
        CREATE OR REPLACE TABLE {gold}
        AS
                select 
                MAKE_DATE(CAST(a.anio AS INT), CAST(a.mes AS INT), 1) AS fecha,
                a.codigo_dane,
                a.municipio,
                a.departamento,
                a.anio,
                a.mes,
                a.total_tematica,
                a.tematica_mes_anterior,
                ROUND(CAST(a.tasa_predicha AS DOUBLE), 2) AS tasa_predicha,
                ROUND(a.tasa_real,2) AS tasa_real,
                ROUND(a.error,2) AS error,
                a.nivel_riesgo_real,
                a.nivel_riesgo_predicho,
                CAST(b.latitud as DOUBLE) AS latitud,
                CAST(b.longitud as DOUBLE) AS longitud,
                b.pres_grup_armados,
                b.pdet_zomac, 
                b.riesgo_victimizacion,
                b.municipio_isla_area_no_municipalizada
                from 
                {rate} a 
                LEFT JOIN {consolidado}  b on a.codigo_dane =b.codigo_municipio
                """



    def build_metrics_query(self):
        rate = f"{self.catalog}.{self.schema}.{self.rate_table}"
        return f"""
            SELECT
                COUNT(*)            AS n,
                AVG(abs_error)      AS mae,
                AVG(sq_error)       AS mse,
                SQRT(AVG(sq_error)) AS rmse,
                AVG(ape_pct)        AS mape_pct,
                AVG(error)          AS sesgo
            FROM {rate}
        """


    def create_rate_table(self):
        print("=" * 50)
        print(f"TASAS - PREDICCIÓN vs REAL ({self.anio})")
        print("=" * 50)

        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, self.consolidate_rate_table())

        print(f" TABLA Tasas consolidado creada ...")

        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, self.generate_threshold_table())

        result = self.sql_executor.execute_sql_query(
           self.catalog, self.schema, self.build_metrics_query()
        )

        metrics = result["data"][0] if result["data"] else {}
        print(f"\n--- Métricas de error {self.anio} ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, self.generate_gold_table())
        return metrics

    def validate_shrinkage_table(self):
        table = f"{self.catalog}.{self.schema}.thresholds_shrinkage"
        query = f"""
        SELECT COUNT(*) AS n_municipios,
        SUM(CASE WHEN p_25 > p_50 OR p_50 > p_75 THEN 1 ELSE 0 END) AS fuera_orden,
        SUM(CASE WHEN casos_totales > 0 AND p_75 = 0 THEN 1 ELSE 0 END) AS p75_cero,
        AVG(shrinkage_factor) AS shrinkage_promedio,
        MIN(p_25) AS min_p25, MAX(p_75) AS max_p75,
        COUNT(CASE WHEN shrinkage_factor < 0.5 THEN 1 END) AS n_high_shrinkage
        FROM {table}
        """
        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        print(f"\nValidación {table}: {result['data']}")
        return result

    def validate_table(self):
        rate = f"{self.catalog}.{self.schema}.{self.rate_table}"
        query = f"SELECT COUNT(*) AS total_rows FROM {rate}"
        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        print(f"\nValidación {rate}: {result['data']}")
        return result

    def run_shrinkage(self, compute_in_python):
        print("=" * 50)
        print(f"THRESHOLDS BAYES SHRINKAGE ({self.anio})")
        print("=" * 50)

        if compute_in_python:
            df_thresh = self.calculate_shrinkage_threshold()
            self.write_shrinkage_thresholds(df_thresh)
        else:
            result = self.sql_executor.execute_sql_query(
                self.catalog, self.schema, self.generate_threshold_shrinkage_table()
            )
            print("Tabla thresholds_shrinkage creada (SQL aprox)")
    
        self.validate_shrinkage_table()
        print("\n" + "=" * 50)
        print("SHRINKAGE THRESHOLDS COMPLETADO")
        print("=" * 50)

    def run(self):
        self.create_rate_table()
        self.validate_table()
        print("\n" + "=" * 50)
        print("CONSOLIDACIÓN COMPLETA")
        print("=" * 50)

    @staticmethod
    def _sql_literal(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(v)
        return "'" + str(v).replace("'", "''") + "'"

def run_consolidated():
    Consolidated().run_shrinkage(compute_in_python=False)
    Consolidated().run()


if __name__ == "__main__":
    run_consolidated()