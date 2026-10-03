from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class Consolidated:
  
    def __init__(self):
        self.config =Config()
        self.sql_executor= SqlExecutor()
        self.catalog = self.config.CATALOG
        self.schema = self.config.SCHEMA
        self.prediction_table = self.config.PREDICTION_TABLE_NAME
        self.consolidated_table = self.config.CONSOLIDATED_TABLE_NAME
        self.silver_table = self.config.TABLE_SILVER_NAME
        self.rate_table = 'hurtos_personas_consolidado'
        self.gold_table = 'hurtos_personas_gold'
        self.anio =2025

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
            WHERE anio BETWEEN 2021 AND 2024
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
        
        


   
    def consolidate_rate_table(self):

        rate = f"{self.catalog}.{self.schema}.{self.rate_table}"
        prediction = f"{self.catalog}.{self.schema}.{self.prediction_table}"
        silver = f"{self.catalog}.{self.schema}.{self.silver_table}"
        consolidado = f"{self.catalog}.{self.schema}.{self.consolidated_table}"
        threshold =f"{self.catalog}.{self.schema}.thresholds"

        return f"""
            CREATE OR REPLACE TABLE {rate}
            USING DELTA 
            AS
            WITH real_{self.anio} AS (
                SELECT
                    a.codigo_dane,
                    a.mes,
                    a.anio,
                    a.total_tematica,
                    a.tematica_mes_anterior,
                    c.pob_dane_2025,
                    CAST((a.total_tematica / c.pob_dane_2025) * 100000 AS DOUBLE) AS tasa_real
                FROM {silver} a
                INNER JOIN {consolidado} c
                    ON a.codigo_dane = c.codigo_municipio
                WHERE a.anio >= {self.anio}
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
                    WHEN t.p_25 IS NULL THEN 'SIN_UMBRAL'
                    WHEN r.tasa_real < t.p_25 THEN 'BAJO'
                    WHEN r.tasa_real < t.p_75 THEN 'MEDIO'      
                    ELSE 'ALTO'
                END AS nivel_riesgo_real,
            CASE
                    WHEN t.p_25 IS NULL THEN 'SIN_UMBRAL'
                    WHEN p.tasa_predicha < t.p_25 THEN 'BAJO'
                    WHEN p.tasa_predicha < t.p_75 THEN 'MEDIO'  
                    ELSE 'ALTO'
            END AS nivel_riesgo_predicho,
                t.p_25,
                t.p_75
            FROM {prediction} p
            LEFT JOIN real_{self.anio} r
                ON CAST(p.codigo_dane AS STRING) = CAST(r.codigo_dane AS STRING)
                AND CAST(p.mes AS INT)  = CAST(r.mes AS INT)
                AND CAST(p.anio AS INT) = CAST(r.anio AS INT)
            LEFT JOIN {threshold} t 
            ON LPAD(CAST(p.codigo_dane AS STRING), 5, '0') = LPAD(CAST(t.codigo_dane AS STRING), 5, '0')

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

    def validate_table(self):
        rate = f"{self.catalog}.{self.schema}.{self.rate_table}"
        query = f"SELECT COUNT(*) AS total_rows FROM {rate}"
        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        print(f"\nValidación {rate}: {result['data']}")
        return result

    def run(self):
        self.create_rate_table()
        self.validate_table()
        print("\n" + "=" * 50)
        print("CONSOLIDACIÓN COMPLETA")
        print("=" * 50)


def run_consolidated():
    Consolidated().run()


if __name__ == "__main__":
    run_consolidated()