import json
import requests
from typing import Optional
from datetime import datetime

from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class Client_Api:
    def __init__(self):
        self.sesion =requests.Session()
        self.config =Config()
        self.client =DataBricksClient().get_client()
        self.sql_executor=SqlExecutor()

    def fetch_data(self, url:str, timeout:int, params: list=None, headers: list=None):
        try:
            response =self.sesion.get(
                url,
                params=params, 
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except ValueError as e:
            raise Exception(f"Invalid JSON response: {e}")

    def create_schema(self, catalog, schema):
        full_name = f"{catalog}.{schema}"
        print(f" Verificate schema {full_name}...")
        try:
            self.client.schemas.get(full_name=full_name)
            print("Schema {schema} exist")
        except Exception as e:
            if "does not exist." in str(e):
                print(f"  Creating schema {schema}...")
                self.client.schemas.create(catalog_name=catalog, name=schema)
                print(f"  Schema {schema} created")
            else:
                raise
    
    def save_json_as_delta(self, data: dict, catalog: str, schema: str, table: str):
        table_full_name = f"{catalog}.{schema}.{table}"

        holidays_data = data.get('holidays', [])

        if not holidays_data:
            raise ValueError("No holidays found in 'holidays' field")

        print(f"  Processing {len(holidays_data)} holidays...")

        values_list = []
        for field in holidays_data:
            values = (
                f"'{field.get('date', '')}'",
                f"'{field.get('name', '')}'",
                f"'{field.get('source', '')}'",
                f"'{field.get('sourceUrl', '')}'",
                f"'{field.get('scope', '')}'",
                f"'{field.get('rule', '')}'"
            )
            values_list.append(f"({', '.join(values)})")

        values_clause = ",\n".join(values_list)
        query = f"""
            INSERT INTO {table_full_name}
            VALUES {values_clause}
        """

        print(f"  Executing batch INSERT...")
        result = self.sql_executor.execute_sql_query(catalog, schema, query)
        print(f"  Insert completed")

        return result


    def create_holidays_table(self, catalog: str, schema: str, table: str = "holidays"):
        
        full_table_name = f"{catalog}.{schema}.{table}"
        
        query = f"""
            CREATE OR REPLACE TABLE  {full_table_name} (
                date DATE COMMENT 'fecha',
                name STRING COMMENT 'nombre del festivo',
                source STRING COMMENT 'Recurso que soporta la creación del festivo',
                source_url STRING COMMENT 'Total de holidays en el JSON',
                scope STRING COMMENT 'Tipo de festivo',
                rule STRING
            )
            USING DELTA
        """
        
        print(f"Creating table {full_table_name}...")
        result = self.sql_executor.execute_sql_query(catalog, schema, query)
        print(f"  Table created successfully")
        return result

    def validate_table(self, catalog: str, schema: str, table: str) -> dict:
        full_table_name = f"{catalog}.{schema}.{table}"
        query = f"SELECT COUNT(*) AS total_rows FROM {full_table_name}"
        
        print(f"Validating table {full_table_name}...")
        result = self.sql_executor.execute_sql_query(catalog, schema, query)
        print(f"  Validation result: {result['data']}")
        return result

 
    


    
        