import re
import unicodedata
from pathlib import Path

import pandas as pd

from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class DataLoader:

    def __init__(self):
        self.config = Config()
        self.client = DataBricksClient().get_client()
        self.sql_executor = SqlExecutor()
        self.data_dir = Path("data")

    def normalize_column_name(self, name: str) -> str:
        name = unicodedata.normalize("NFD", name)
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        name = name.lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    def ensure_catalog(self):
        catalog_name = self.config.CATALOG
        print(f"Verifying catalog {catalog_name}...")
        try:
            self.client.catalogs.get(name=catalog_name)
            print(f"  Catalog {catalog_name} exists")
        except Exception as e:
            if "does not exist." in str(e):
                print(f"  Creating catalog {catalog_name}...")
                self.client.catalogs.create(name=catalog_name)
                print(f"  Catalog {catalog_name} created")
            else:
                raise

    def ensure_schema(self):
        catalog_name = self.config.CATALOG
        schema_name = self.config.SCHEMA
        full_name = f"{catalog_name}.{schema_name}"
        print(f"Verifying schema {full_name}...")
        try:
            self.client.schemas.get(full_name=full_name)
            print(f"  Schema {schema_name} exists")
        except Exception as e:
            if "does not exist." in str(e):
                print(f"  Creating schema {schema_name}...")
                self.client.schemas.create(catalog_name=catalog_name, name=schema_name)
                print(f"  Schema {schema_name} created")
            else:
                raise

    def ensure_volume(self):
        catalog_name = self.config.CATALOG
        schema_name = self.config.SCHEMA
        volume_name = self.config.VOLUME_NAME
        full_name = f"{catalog_name}.{schema_name}.{volume_name}"
        try:
            volume = self.client.volumes.read(name=full_name)
            print(f"  Volume {volume_name} exists")
            return volume
        except Exception as e:
            if "does not exist." in str(e):
                print(f"  Creating volume {volume_name}...")
                from databricks.sdk.service.catalog import VolumeType

                volume = self.client.volumes.create(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    name=volume_name,
                    volume_type=VolumeType.MANAGED,
                )
                print(f"  Volume {volume_name} created")
                return volume
            else:
                raise

    def convert_to_parquet(self, file_path: Path) -> Path:
        parquet_path = file_path.with_suffix(".parquet")

        print(f"  Reading {file_path.name}...")
        if file_path.suffix == ".xlsx":
            df = pd.read_excel(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        df.columns = [self.normalize_column_name(col) for col in df.columns]
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved as {parquet_path.name} with columns: {list(df.columns)}")
        return parquet_path

    def upload_to_volume(self, local_path: Path) -> str:
        volume = self.ensure_volume()
        volume_path = (
            f"/Volumes/{volume.catalog_name}/{volume.schema_name}/{volume.name}"
            f"/files/{local_path.name}"
        )
        print(f"  Uploading to {volume_path}...")
        with open(local_path, "rb") as f:
            self.client.files.upload(volume_path, f, overwrite=True)
        print(f"  File uploaded")
        return volume_path

    def create_table(self, volume_path: str, table_name: str):
        catalog = self.config.CATALOG
        schema = self.config.SCHEMA
        full_table = f"{catalog}.{schema}.{table_name}"

        query = f"""
            CREATE OR REPLACE TABLE {full_table}
            USING DELTA
            AS SELECT * FROM parquet.`{volume_path}`
        """
        print(f"  Creating table {full_table}...")
        self.sql_executor.execute_sql_query(catalog, schema, query)
        print(f"  Table {full_table} created")

    def validate_table(self, table_name: str):
        catalog = self.config.CATALOG
        schema = self.config.SCHEMA
        full_table = f"{catalog}.{schema}.{table_name}"
        query = f"SELECT COUNT(*) AS total_rows FROM {full_table}"

        result = self.sql_executor.execute_sql_query(catalog, schema, query)
        print(f"  Validation: {result['data']}")
        return result

    def load_file(self, file_path: Path):
        print(f"\nProcessing: {file_path.name}")

        parquet_path = self.convert_to_parquet(file_path)
        volume_path = self.upload_to_volume(parquet_path)
        table_name = file_path.stem.lower()

        self.create_table(volume_path, table_name)
        self.validate_table(table_name)

    def run(self):
        print("=" * 50)
        print("DATA LOADER - LOAD ALL FILES")
        print("=" * 50)

        self.ensure_catalog()
        self.ensure_schema()

        files = []
        for ext in ["*.xlsx", "*.csv", "*.parquet"]:
            files.extend(self.data_dir.glob(ext))

        files = sorted(set(files))

        if not files:
            print("No files found in data/")
            return

        print(f"Found {len(files)} files: {[f.name for f in files]}")

        for file_path in files:
            self.load_file(file_path)

        print("=" * 50)
        print("DATA LOADER COMPLETE")
        print("=" * 50)


if __name__ == "__main__":
    loader = DataLoader()
    loader.run()
