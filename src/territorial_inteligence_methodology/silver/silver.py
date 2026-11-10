from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor
from territorial_inteligence_methodology.utils.domains import STRUCTURE_SILVER_TABLE, STRUCTURE_SILVER_TABLE_MUNICIPALITY


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
        silver_table =f"{self.config.CATALOG}.{self.config.SCHEMA}.{self.config.TABLE_SILVER_NAME}"
        query = f"""
            CREATE OR REPLACE TABLE {silver_table}
            USING DELTA
            AS {STRUCTURE_SILVER_TABLE}
        """
        print(f"Creating silver table {silver_table}...")
        self.sql_executor.execute_sql_query(
            self.config.CATALOG,
            self.config.SCHEMA,
            query
        )

    def generate_silver_municipality_table(self):
            silver_table =f"{self.config.CATALOG}.{self.config.SCHEMA}.{self.config.TABLE_SILVER_MUNICIPALITY_NAME}"
            query = f"""
                CREATE OR REPLACE TABLE {silver_table}
                USING DELTA
                AS {STRUCTURE_SILVER_TABLE_MUNICIPALITY}
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
    
    def validate_silver_municipality_table(self):
        table_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.{self.config.TABLE_SILVER_MUNICIPALITY_NAME}"
        query = f"""SELECT COUNT(*) AS filas,
       COUNT(DISTINCT CONCAT(CAST(codigo_dane AS STRING), '-', id_barrio)) AS barrios,
       SUM(hay_observacion) AS celdas_con_datos,
       SUM(CASE WHEN hay_observacion = 1 THEN total_tematica ELSE 0 END) AS eventos
        FROM {table_name}"""

        print(f"Validating table {table_name}...")
        result = self.sql_executor.execute_sql_query(
            self.config.CATALOG,
            self.config.SCHEMA,
            query
        )

        return result

    def execution_silver_table(self, enable_municipality:bool):
        print("=" * 50)
        print("SILVER LAYER - TRANSFORMATION PROCESS")
        print("=" * 50)
        
        self.generate_silver_table()
        self.validate_silver_table()

        if(enable_municipality):
            self.generate_silver_municipality_table()
            self.validate_silver_municipality_table()


        print("=" * 50)
        print("SILVER LAYER - SILVER TABLE GENERATED")
        print("=" * 50)