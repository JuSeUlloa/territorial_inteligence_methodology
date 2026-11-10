
from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor

class Services:
    
    
    def __init__(self, ttl_seconds:int):
     self.config = Config()
     self.sql_executor = SqlExecutor()
     self.catalog = self.config.CATALOG
     self.schema = self.config.SCHEMA
     self.ttl= ttl_seconds


    def get_gold_consolidated_table(self):
        table = "hurtos_personas_consolidado"
        consolidated_table = f"{self.catalog}.{self.schema}.{table}"
        query = f"""
        SELECT
            municipio,
            codigo_dane,
            departamento,
            mes,
            anio,
            tasa_predicha,
            tasa_real,
            total_tematica,
            tematica_mes_anterior,
            pob_dane_2025,
            error,
            abs_error,
            sq_error,
            ape_pct,
            nivel_riesgo_real,
            nivel_riesgo_predicho,
            p_25,
            p_50,
            p_75
        FROM {consolidated_table}
        WHERE  mes <> ( SELECT MAX(mes) FROM {consolidated_table})
        """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]
    
    def get_gold_consolidated_month_table(self, mes:int):
        table = "hurtos_personas_consolidado"
        consolidated_table = f"{self.catalog}.{self.schema}.{table}"
        query = f"""
        SELECT
            municipio,
            codigo_dane,
            departamento,
            mes,
            anio,
            tasa_predicha,
            tasa_real,
            total_tematica,
            tematica_mes_anterior,
            pob_dane_2025,
            error,
            abs_error,
            sq_error,
            ape_pct,
            nivel_riesgo_real,
            nivel_riesgo_predicho,
            p_25,
            p_50,
            p_75
        FROM {consolidated_table}
        WHERE  mes = {mes}
        """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]
    
    def get_prediction_table(self):
        table = "hurtos_personas_consolidado"
        consolidated_table = f"{self.catalog}.{self.schema}.{table}"
        consolidated = f"{self.catalog}.{self.schema}.consolidado"
        query = f"""
            SELECT
            a.municipio,
            a.codigo_dane,
            a.departamento,
            a.mes,
            a.anio,
            a.tasa_predicha,
            a.tasa_real,
            round(a.tasa_predicha * b.pob_dane_2025 / 100000) as total_tematica,
            a.tematica_mes_anterior,
            a.pob_dane_2025,
            a.error,
            a.abs_error,
            a.sq_error,
            a.ape_pct,
            a.nivel_riesgo_real,
            a.nivel_riesgo_predicho,
            a.p_25,
            a.p_50,
            a.p_75
        FROM {consolidated_table} a
        JOIN {consolidated} b on a.codigo_dane=b.codigo_municipio
        WHERE  a.mes = (SELECT MAX(mes) FROM {consolidated_table})
        """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]
    
    def get_prediction_municipality_table(self):
        consolidated_table = f"{self.catalog}.{self.schema}.hurto_personas_gold_municipio"
        query = f"""
            SELECT
            a.id_barrio,
            a.barrio,
            a.municipio,
            a.codigo_dane,
            null as departamento,
            a.mes,
            a.anio,
            a.tasa_predicha,
            a.tasa_real,
            round(a.tasa_predicha * a.pob_dane_2025 / 100000) as total_tematica,
            a.tematica_mes_anterior,
            a.pob_dane_2025,
            a.error,
            a.abs_error,
            a.sq_error,
            null as ape_pct,
            a.nivel_riesgo_real,
            a.nivel_riesgo_predicho,
            a.p_25,
            a.p_50,
            a.p_75
        FROM {consolidated_table} a
        """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]
    
    def get_prediction_municipality_current_table(self):
        consolidated_table = f"{self.catalog}.{self.schema}.hurto_personas_gold_municipio"
        query = f"""
            SELECT
            a.id_barrio,
            a.barrio,
            a.municipio,
            a.codigo_dane,
            null as departamento,
            a.mes,
            a.anio,
            a.tasa_predicha,
            a.tasa_real,
            round(a.tasa_predicha * a.pob_dane_2025 / 100000) as total_tematica,
            a.tematica_mes_anterior,
            a.pob_dane_2025,
            a.error,
            a.abs_error,
            a.sq_error,
            null as ape_pct,
            a.nivel_riesgo_real,
            a.nivel_riesgo_predicho,
            a.p_25,
            a.p_50,
            a.p_75
        FROM {consolidated_table} a
        WHERE  a.anio = (SELECT MAX(anio) FROM {consolidated_table})
        """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]
    
    def get_prediction_municipality_month_table(self, month:int):
        consolidated_table = f"{self.catalog}.{self.schema}.hurto_personas_gold_municipio"
        query = f"""
            SELECT
            a.id_barrio,
            a.barrio,
            a.municipio,
            a.codigo_dane,
            null as departamento,
            a.mes,
            a.anio,
            a.tasa_predicha,
            a.tasa_real,
            round(a.tasa_predicha * a.pob_dane_2025 / 100000) as total_tematica,
            a.tematica_mes_anterior,
            a.pob_dane_2025,
            a.error,
            a.abs_error,
            a.sq_error,
            null as ape_pct,
            a.nivel_riesgo_real,
            a.nivel_riesgo_predicho,
            a.p_25,
            a.p_50,
            a.p_75
        FROM {consolidated_table} a
        WHERE  a.mes = {month}
        """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]
    
    def get_bronze_data(self):
        bronze_table = f"{self.catalog}.{self.schema}.hurtos_personas_bronce"
        query = f"""
            SELECT
            tematica,
            anio,
            mes,
            departamento,
            codigo_dane,
            municipio,
            armas_medios,
            agrupa_edad,
            genero,
            zona,
            clase_sitio,
            dia,
            cantidad
            FROM
            {bronze_table}
            LIMIT 1000
        """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]

    def get_municipality_by_code(self, codigo_dane: str):
        codigo = str(codigo_dane).strip()
        if not (codigo.isdigit() and len(codigo) == 5):
            raise ValueError(
                f'codigo_dane invalido: {codigo_dane!r} (se esperaba 5 digitos)'
            )
        municipios_geo = f"{self.catalog}.{self.schema}.municipios_geo"
        query = f"""
                 SELECT
               m.codigo_municipio AS codigo_dane,
               m.municipio,
               m.codigo_departamento,
               m.departamento AS departamento_nombre,
               CASE
                   WHEN ST_IsValid(m.geom_simp) THEN ST_AsGeoJSON(m.geom_simp)
                   ELSE ST_AsGeoJSON(m.geometry)
               END AS geometry
            FROM (
                SELECT codigo_municipio, municipio, codigo_departamento,
                       departamento, geometry,
                       ST_Simplify(geometry, 0.001) AS geom_simp
                FROM {municipios_geo}
                WHERE LPAD(CAST(codigo_municipio AS STRING), 5, '0') = '{codigo}'
            ) m
                """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        if not datos:
            raise ValueError(f'codigo_dane {codigo} no existe en municipios_geo')
        return [{k: v for k, v in row.items()} for row in datos]

    def get_municipalities_by_department(self, codigo_departamento: str):
        codigo = str(codigo_departamento).strip()
        if not (codigo.isdigit() and 1 <= len(codigo) <= 2):
            raise ValueError(
                f'codigo_departamento invalido: {codigo_departamento!r} (se esperaba 1-2 digitos)'
            )

        municipios_geo = f"{self.catalog}.{self.schema}.municipios_geo"
        departamentos_geo = f"{self.catalog}.{self.schema}.departamentos_geo"

        query = f"""
                SELECT
                   LPAD(CAST(m.codigo_municipio AS STRING), 5, '0') AS codigo_dane,
                   m.municipio,
                   LPAD(CAST(m.codigo_departamento AS STRING), 2, '0') AS codigo_departamento,
                   d.departamento_nombre,
                   CASE
                       WHEN ST_IsValid(m.geom_simp) THEN ST_AsGeoJSON(m.geom_simp)
                       ELSE ST_AsGeoJSON(m.geometry)
                   END AS geometry
                FROM (
                    SELECT
                       codigo_municipio,
                       municipio,
                       codigo_departamento,
                       geometry,
                       ST_Simplify(geometry, 0.001)  AS geom_simp
                    FROM {municipios_geo}
                    WHERE LPAD(CAST(codigo_departamento AS STRING), 2, '0') = LPAD('{codigo}', 2, '0')
                ) m
                LEFT JOIN {departamentos_geo} d
                  ON LPAD(CAST(d.codigo_departamento AS STRING), 2, '0')
                   = LPAD(CAST(m.codigo_departamento AS STRING), 2, '0')
                """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]

    def get_barrio_geo(self):
        table = "municipio_barrios_geo"
        barrios_geo = f"{self.catalog}.{self.schema}.{table}"
        barrios_tabla  = f"{self.catalog}.{self.schema}.hurto_personas_municipio"
        query = f"""
               SELECT
                  DISTINCT t.id_barrio,
                   t.barrio,
                   b.codigo_divipola as codigo_dane,
                   CASE
                       WHEN ST_IsValid(t.geom_simp) THEN ST_AsGeoJSON(t.geom_simp)
                       ELSE ST_AsGeoJSON(t.geometry)
                   END AS geometry
                FROM (
                    SELECT
                       id_barrio,
                       barrio,
                       geometry,
                       ST_Simplify(geometry, 0.0001)  AS geom_simp
                    FROM {barrios_geo}
                    WHERE geometry IS NOT NULL
                ) t
                JOIN {barrios_tabla} b using (id_barrio)
                """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]

    def get_department_geo(self):
        table = "departamentos_geo"
        departamento_geo = f"{self.catalog}.{self.schema}.{table}"

        query = f"""
                SELECT
                   codigo_departamento,
                   departamento_nombre,
                   CASE
                       WHEN ST_IsValid(geom_simp) THEN ST_AsGeoJSON(geom_simp)
                       ELSE ST_AsGeoJSON(geometry)
                   END AS geometry
                FROM (
                    SELECT
                       codigo_departamento,
                       departamento_nombre,
                       geometry,
                       ST_Simplify(geometry, 0.01)  AS geom_simp
                    FROM {departamento_geo}
                ) t
                """
        datos = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)['data']
        return [{k: v for k, v in row.items()} for row in datos]


