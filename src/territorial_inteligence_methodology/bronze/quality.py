from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor
from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.bronze.schema import SchemaGenerator


class Quality:
    def __init__(self):
        self.client = DataBricksClient().get_client()
        self.sql_executor = SqlExecutor()
        self.config = Config()
        self.schema_generator = SchemaGenerator()
    
    def _get_table_full_name(self, table_name: str) -> str:
        return f"{self.config.CATALOG}.{self.config.SCHEMA}.{table_name}"
    
    def get_row_count(self, table_name: str) -> int:
        full_name = self._get_table_full_name(table_name)
        query = f"SELECT COUNT(*) AS row_count FROM {full_name}"
        result = self.sql_executor.execute_sql_query(
            catalog_name=self.config.CATALOG,
            schema_name=self.config.SCHEMA,
            query=query
        )
        return int(result["data"][0]["row_count"])
    
    def get_nulls_count(self, table_name: str) -> dict:
        full_name = self._get_table_full_name(table_name)
        columns = self.schema_generator.get_columns()
        
        null_expressions = ", ".join(
            [f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}" for col in columns]
        )
        query = f"SELECT {null_expressions} FROM {full_name}"
        
        result = self.sql_executor.execute_sql_query(
            catalog_name=self.config.CATALOG,
            schema_name=self.config.SCHEMA,
            query=query
        )
        return result["data"][0] if result["data"] else {}
    
    def get_distinct_count(self, table_name: str) -> dict:
        full_name = self._get_table_full_name(table_name)
        columns = self.schema_generator.get_columns()
        
        distinct_expressions = ", ".join(
            [f"COUNT(DISTINCT {col}) AS {col}" for col in columns]
        )
        query = f"SELECT {distinct_expressions} FROM {full_name}"
        
        result = self.sql_executor.execute_sql_query(
            catalog_name=self.config.CATALOG,
            schema_name=self.config.SCHEMA,
            query=query
        )
        return result["data"][0] if result["data"] else {}
    
    def get_duplicate_count(self, table_name: str) -> dict:
        full_name = self._get_table_full_name(table_name)
        columns = self.schema_generator.get_columns()
        columns_str = ', '.join(columns)
        
        query = f"""
            SELECT COUNT(*) AS duplicated_rows
            FROM (
                SELECT {columns_str}, COUNT(*) AS cnt
                FROM {full_name}
                GROUP BY {columns_str}
                HAVING cnt > 1
            )
        """
        
        result = self.sql_executor.execute_sql_query(
            catalog_name=self.config.CATALOG,
            schema_name=self.config.SCHEMA,
            query=query
        )
        return result["data"][0] if result["data"] else {}
    
    def get_columns_stats(self, table_name: str, column: str) -> dict:
        full_name = self._get_table_full_name(table_name)
        
        query = f"""
            SELECT
                MIN({column}) AS min_value,
                MAX({column}) AS max_value,
                AVG({column}) AS avg_value,
                STDDEV({column}) AS stddev_value
            FROM {full_name}
        """
        
        result = self.sql_executor.execute_sql_query(
            catalog_name=self.config.CATALOG,
            schema_name=self.config.SCHEMA,
            query=query
        )
        return result["data"][0] if result["data"] else {}
    
    def execute_quality_checks(self, table_name: str):
        table_name= table_name or self.config.TABLE_NAME
        full_name = self._get_table_full_name(table_name)
        
        print("=" * 50)
        print(f"DATA QUALITY REPORT: {full_name}")
        print("=" * 50)
        
        row_count = self.get_row_count(table_name)
        print(f"\nTotal rows: {row_count:,}")
        
        print(f"\n--- Null counts ---")
        nulls = self.get_nulls_count(table_name)
        for col, count in nulls.items():
            print(f"  {col}: {int(count):,}")
        
        print(f"\n--- Distinct values ---")
        distincts = self.get_distinct_count(table_name)
        for col, count in distincts.items():
            print(f"  {col}: {int(count):,}")
        
        print(f"\n--- Duplicates ---")
        dups = self.get_duplicate_count(table_name)
        print(f"  Duplicated rows: {int(dups.get('duplicated_rows', 0)):,}")
        
        print(f"\n--- Numeric column stats ---")
        for col in ["anio", "cantidad"]:
            if col in self.schema_generator.get_columns():
                stats = self.get_columns_stats(table_name, col)
                print(f"  {col}: min={stats.get('min_value')}, max={stats.get('max_value')}, avg={stats.get('avg_value')}, stddev={stats.get('stddev_value')}")
        
        print("\n" + "=" * 50)
