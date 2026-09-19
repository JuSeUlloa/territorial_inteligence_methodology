from pathlib import Path
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor
from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.bronze.schema import SchemaGenerator


class ParquetLoader:
    
    
    def __init__(self):
        self.client = DataBricksClient().get_client()
        self.sql_executor = SqlExecutor()
        self.config = Config()
        self.schema_generator = SchemaGenerator()
    
    def ensure_catalog(self):
        catalog_name = self.config.CATALOG
        print(f'Verifying catalog {catalog_name}...')
        try:
            self.client.catalogs.get(name=catalog_name)
            print(f'  Catalog {catalog_name} exists')
        except Exception as e:
            if "does not exist." in str(e):
                print(f'  Creating catalog {catalog_name}...')
                self.client.catalogs.create(name=catalog_name)
                print(f'  Catalog {catalog_name} created')
            else:
                raise
    
    def ensure_schema(self):
        catalog_name = self.config.CATALOG
        schema_name = self.config.SCHEMA
        full_name = f'{catalog_name}.{schema_name}'
        print(f'Verifying schema {full_name}...')
        try:
            self.client.schemas.get(full_name=full_name)
            print(f'  Schema {schema_name} exists')
        except Exception as e:
            if "does not exist." in str(e):
                print(f'  Creating schema {schema_name}...')
                self.client.schemas.create(catalog_name=catalog_name, name=schema_name)
                print(f'  Schema {schema_name} created')
            else:
                raise
    
    def ensure_volume(self, volume_name: str):
        catalog_name = self.config.CATALOG
        schema_name = self.config.SCHEMA
        full_name = f'{catalog_name}.{schema_name}.{volume_name}'
        try:
            volume = self.client.volumes.read(name=full_name)
            print(f'  Volume {volume_name} exists')
            return volume
        except Exception as e:
            if "does not exist." in str(e):
                print(f'  Creating volume {volume_name}...')
                volume = self.client.volumes.create(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    name=volume_name,
                    volume_type=VolumeType.MANAGED
                )
                print(f'  Volume {volume_name} created')
                return volume
            else:
                raise
    
    
    def create_delta_table(self, volume_path: str, table_name: str):
        catalog_name = self.config.CATALOG
        schema_name = self.config.SCHEMA
        full_table_name = f'{catalog_name}.{schema_name}.{table_name}'
        print()
        source = f"parquet.`{volume_path}`"
        select_query = self.schema_generator.build_select_query(source)
        query = f"""
            CREATE OR REPLACE TABLE {full_table_name}
            USING DELTA
            AS {select_query}
        """
        print(f'Creating table {full_table_name} from {volume_path}...')
        self.sql_executor.execute_sql_query(catalog_name, schema_name, query)
        print(f'  Table {full_table_name} created successfully')
    
    def validate_data_table(self, table_name: str) -> dict:
        full_table_name = f"{self.config.CATALOG}.{self.config.SCHEMA}.{table_name}"
        query = f"SELECT COUNT(*) AS total_rows FROM {full_table_name}"
        
        print(f'Validating data in table {full_table_name}...')
        result = self.sql_executor.execute_sql_query(self.config.CATALOG, self.config.SCHEMA, query)
        print(f'  Validation result: {result["data"]}')
        return result
    
    def load_table(self, local_path: str = None, volume_name: str = None, table_name: str = None):
        local_path = local_path or f"data/{self.config.FILE_NAME}"
        volume_name = volume_name or self.config.VOLUME_NAME
        table_name = table_name or self.config.TABLE_NAME
        print(f"------------------- {table_name}")
        
        print("=" * 50)
        print("BRONZE LAYER - LOAD PROCESS")
        print("=" * 50)
        
        self.ensure_catalog()
        self.ensure_schema()
        
        volume_path = self.upload_parquet_to_volume(local_path, volume_name)
        
        self.create_delta_table(volume_path, table_name)
        
        self.validate_data_table(table_name)
        
        print("=" * 50)
        print(f"BRONZE LOAD COMPLETE: {self.config.CATALOG}.{self.config.SCHEMA}.{table_name}")
        print("=" * 50)
