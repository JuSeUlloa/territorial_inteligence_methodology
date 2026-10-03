from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class SilverEncoder:

    def __init__(self):
        self.config =Config()
        self.client =DataBricksClient()
        self.sql_executor =SqlExecutor()
        self.CATALOG =self.config.CATALOG
        self.SCHEMA =self.config.SCHEMA
        self.TABLE_BRONCE_NAME =self.config.TABLE_BRONZE_NAME
        self.TABLE_SILVER_NAME =self.config.TABLE_SILVER_NAME
        self.consolidado =self.config.CONSOLIDATED_TABLE_NAME
        

    
    def generate_silver_table(self):
        source_table =f"{self.CATALOG}.{self.SCHEMA}.{self.TABLE_BRONCE_NAME}"
        silver_table =f"{self.config.CATALOG}.{self.config.SCHEMA}.{self.config.TABLE_SILVER_NAME}"
               
        query = f"""
            CREATE OR REPLACE TABLE {silver_table}
            USING DELTA
            AS
            WITH agregation as (
            SELECT 
                            municipio,
                            departamento,
                            LPAD(LEFT(codigo_dane, LENGTH(codigo_dane) - 3), 5, '0') as codigo_dane,
                            mes,
                            anio,
                            SUM(cantidad) as total_tematica,
                            SUM(CASE WHEN zona = 3 THEN cantidad ELSE 0 END) as tematica_zona_urbana,
                            SUM(CASE WHEN zona = 2 THEN cantidad ELSE 0 END) as tematica_zona_rural,
                            SUM(CASE WHEN zona = 1 THEN cantidad ELSE 0 END) as tematica_zona_otras,
                            SUM(CASE WHEN armas_medios = 3 THEN cantidad ELSE 0 END) as tematica_arma_fuego,
                            SUM(CASE WHEN armas_medios = 2 THEN cantidad ELSE 0 END) as tematica_arma_blanca,
                            SUM(CASE WHEN armas_medios = 4 THEN cantidad ELSE 0 END) as tematica_contundentes,
                            SUM(CASE WHEN genero = 2 THEN cantidad ELSE 0 END) as tematica_masculino,
                            SUM(CASE WHEN genero = 1 THEN cantidad ELSE 0 END) as tematica_femenino,
                            SUM(CASE WHEN agrupa_edad = 2 THEN cantidad ELSE 0 END) as tematica_adultos,
                            SUM(CASE WHEN agrupa_edad = 3 THEN cantidad ELSE 0 END) as tematica_jovenes,
                            SUM(CASE WHEN agrupa_edad = 4 THEN cantidad ELSE 0 END) as tematica_menores,
                            SUM(CASE WHEN dia = 'lun.' THEN cantidad ELSE 0 END) as tematica_lunes,
                            SUM(CASE WHEN dia = 'mar.' THEN cantidad ELSE 0 END) as tematica_martes,
                            SUM(CASE WHEN dia = 'mié.' THEN cantidad ELSE 0 END) as tematica_miercoles,
                            SUM(CASE WHEN dia = 'jue.' THEN cantidad ELSE 0 END) as tematica_jueves,
                            SUM(CASE WHEN dia = 'vie.' THEN cantidad ELSE 0 END) as tematica_viernes,
                            SUM(CASE WHEN dia = 'sáb.' THEN cantidad ELSE 0 END) as tematica_sabado,
                            SUM(CASE WHEN dia = 'dom.' THEN cantidad ELSE 0 END) as tematica_domingo
                        FROM {source_table}
                        WHERE municipio <> '-'
                        GROUP BY 1,2,3,4,5
                        ),
                        temporal_metrics as (
                        SELECT 
                                        municipio,
                                        codigo_dane,
                                        departamento,
                                        mes,
                                        anio,
                                        total_tematica,
                                        tematica_zona_urbana,
                                        tematica_zona_rural,
                                        tematica_zona_otras,
                                        tematica_arma_fuego,
                                        tematica_arma_blanca,
                                        tematica_contundentes,
                                        tematica_masculino,
                                        tematica_femenino,
                                        tematica_adultos,
                                        tematica_jovenes,
                                        tematica_menores,
                                        tematica_lunes,
                                        tematica_martes,
                                        tematica_miercoles,
                                        tematica_jueves,
                                        tematica_viernes,
                                        tematica_sabado,
                                        tematica_domingo,
                                        LAG(total_tematica, 1) OVER (
                                            PARTITION BY municipio ORDER BY anio, mes
                                        ) as tematica_mes_anterior,
                                        AVG(total_tematica) OVER (
                                            PARTITION BY municipio 
                                            ORDER BY anio, mes 
                                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                                        ) as media_movil_3_meses,
                                        AVG(total_tematica) OVER (
                                            PARTITION BY municipio 
                                            ORDER BY anio, mes 
                                            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                                        ) as media_movil_6_meses,
                                        (6 * SUM(ROW_NUMBER() OVER (PARTITION BY municipio ORDER BY anio, mes) * total_tematica) OVER (
                                            PARTITION BY municipio 
                                            ORDER BY anio, mes 
                                            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                                        ) - SUM(ROW_NUMBER() OVER (PARTITION BY municipio ORDER BY anio, mes)) OVER (
                                            PARTITION BY municipio 
                                            ORDER BY anio, mes 
                                            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                                        ) * SUM(total_tematica) OVER (
                                            PARTITION BY municipio 
                                            ORDER BY anio, mes 
                                            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                                        )) / NULLIF(
                                            6 * SUM(POWER(ROW_NUMBER() OVER (PARTITION BY municipio ORDER BY anio, mes), 2)) OVER (
                                                PARTITION BY municipio 
                                                ORDER BY anio, mes 
                                                ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                                            ) - POWER(SUM(ROW_NUMBER() OVER (PARTITION BY municipio ORDER BY anio, mes)) OVER (
                                                PARTITION BY municipio 
                                                ORDER BY anio, mes 
                                                ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                                            ), 2),
                                            0
                                        ) as tendencia,
                                        CASE 
                                            WHEN LAG(total_tematica, 1) OVER (PARTITION BY municipio ORDER BY anio, mes) > 0 
                                            THEN ((total_tematica - LAG(total_tematica, 1) OVER (PARTITION BY municipio ORDER BY anio, mes)) 
                                                  / LAG(total_tematica, 1) OVER (PARTITION BY municipio ORDER BY anio, mes)) * 100
                                            ELSE NULL 
                                        END as crecimiento_intermensual,
                                        CASE 
                                            WHEN LAG(total_tematica, 12) OVER (PARTITION BY municipio ORDER BY anio, mes) > 0 
                                            THEN ((total_tematica - LAG(total_tematica, 12) OVER (PARTITION BY municipio ORDER BY anio, mes)) 
                                                  / LAG(total_tematica, 12) OVER (PARTITION BY municipio ORDER BY anio, mes)) * 100
                                            ELSE NULL 
                                        END as crecimiento_interanual,
                                        total_tematica - MAX(total_tematica) OVER (
                                            PARTITION BY municipio, mes 
                                            ORDER BY anio 
                                            ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING
                                        ) as max_anio_anterior,
                                        total_tematica - LAG(total_tematica, 1) OVER (
                                            PARTITION BY municipio ORDER BY anio, mes
                                        ) as max_mes_anterior
                                    FROM agregation
                                    ), percentajes as (
                                    SELECT 
                                            municipio,
                                            codigo_dane,
                                            departamento,
                                            mes,
                                            anio,
                                            total_tematica,
                                            tematica_mes_anterior,
                                            media_movil_3_meses,
                                            media_movil_6_meses,
                                            tendencia,
                                            crecimiento_intermensual,
                                            crecimiento_interanual,
                                            max_anio_anterior,
                                            max_mes_anterior,
                                            CASE WHEN total_tematica > 0 THEN (tematica_zona_urbana / total_tematica) * 100 ELSE NULL END as pct_zona_urbana,
                                            CASE WHEN total_tematica > 0 THEN (tematica_zona_rural / total_tematica) * 100 ELSE NULL END as pct_zona_rural,
                                            CASE WHEN total_tematica > 0 THEN (tematica_arma_fuego / total_tematica) * 100 ELSE NULL END as pct_arma_fuego,
                                            CASE WHEN total_tematica > 0 THEN (tematica_arma_blanca / total_tematica) * 100 ELSE NULL END as pct_arma_blanca,
                                            CASE WHEN total_tematica > 0 THEN (tematica_contundentes / total_tematica) * 100 ELSE NULL END as pct_arma_contundente,
                                            CASE WHEN total_tematica > 0 THEN (tematica_masculino / total_tematica) * 100 ELSE NULL END as pct_genero_masculino,
                                            CASE WHEN total_tematica > 0 THEN (tematica_femenino / total_tematica) * 100 ELSE NULL END as pct_genero_femenino,
                                            CASE WHEN total_tematica > 0 THEN (tematica_adultos / total_tematica) * 100 ELSE NULL END as pct_edad_adultos,
                                            CASE WHEN total_tematica > 0 THEN (tematica_jovenes / total_tematica) * 100 ELSE NULL END as pct_edad_jovenes,
                                            CASE WHEN total_tematica > 0 THEN (tematica_menores / total_tematica) * 100 ELSE NULL END as pct_edad_menores,
                                            CASE WHEN total_tematica > 0 THEN (tematica_lunes / total_tematica) * 100 ELSE NULL END as pct_dia_lunes,
                                            CASE WHEN total_tematica > 0 THEN (tematica_martes / total_tematica) * 100 ELSE NULL END as pct_dia_martes,
                                            CASE WHEN total_tematica > 0 THEN (tematica_miercoles / total_tematica) * 100 ELSE NULL END as pct_dia_miercoles,
                                            CASE WHEN total_tematica > 0 THEN (tematica_jueves / total_tematica) * 100 ELSE NULL END as pct_dia_jueves,
                                            CASE WHEN total_tematica > 0 THEN (tematica_viernes / total_tematica) * 100 ELSE NULL END as pct_dia_viernes,
                                            CASE WHEN total_tematica > 0 THEN (tematica_sabado / total_tematica) * 100 ELSE NULL END as pct_dia_sabado,
                                            CASE WHEN total_tematica > 0 THEN (tematica_domingo / total_tematica) * 100 ELSE NULL END as pct_dia_domingo                
                                        FROM temporal_metrics
                                    )
                                    SELECT 
                                        municipio,
                                        codigo_dane,
                                        departamento,
                                        mes,
                                        anio,
                                        total_tematica,
                                        tematica_mes_anterior,
                                        media_movil_3_meses,
                                        media_movil_6_meses,
                                        tendencia,
                                        crecimiento_intermensual,
                                        crecimiento_interanual,
                                        max_anio_anterior,
                                        max_mes_anterior,
                                        pct_zona_urbana,
                                        pct_zona_rural,
                                        pct_arma_fuego,
                                        pct_arma_blanca,
                                        pct_arma_contundente,
                                        pct_genero_masculino,
                                        pct_genero_femenino,
                                        pct_edad_adultos,
                                        pct_edad_jovenes,
                                        pct_edad_menores,
                                        pct_dia_lunes,
                                        pct_dia_martes,
                                        pct_dia_miercoles,
                                        pct_dia_jueves,
                                        pct_dia_viernes,
                                        pct_dia_sabado,
                                        pct_dia_domingo
                                        FROM percentajes
                                        ORDER BY municipio, anio, mes
        """

            

        print(f"Creating silver table {silver_table}...")
        self.sql_executor.execute_sql_query(
            self.config.CATALOG,
            self.config.SCHEMA,
            query
        )
        
    
    def validate_silver_table(self):
        table_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.{self.config.TABLE_SILVER_NAME}"
        query = f"SELECT COUNT(*) AS total_rows FROM {table_name}"

        print(f"Validating table {table_name}...")
        result = self.sql_executor.execute_sql_query(
            self.config.CATALOG,
            self.config.SCHEMA,
            query
        )

        return result

    def execution_silver_table(self):
        print("=" * 50)
        print("SILVER LAYER - TRANSFORMATION PROCESS")
        print("=" * 50)

        self.generate_silver_table()


        print("=" * 50)
        print("SILVER LAYER - SILVER TABLE GENERATED")
        print("=" * 50)