from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor
from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.domains import FIELDS_NOT_ENCODED, FIELDS_TO_ENCODE, MONTH_MAPPING


class BronzeEncoder:

    def __init__(self):
        self.client = DataBricksClient().get_client()
        self.sql_executor = SqlExecutor()
        self.config = Config()

    def get_unique_values(self, table_name: str, column: str) -> list:
        query = f"SELECT DISTINCT {column} FROM {table_name} WHERE {column} IS NOT NULL ORDER BY {column}"
        result = self.sql_executor.execute_sql_query(
            self.config.CATALOG,
            self.config.SCHEMA,
            query
        )
        return [row[column] for row in result['data']]

    def create_encoding_mapping(self, unique_values: list) -> dict:
        return {value: idx + 1 for idx, value in enumerate(unique_values)}

    def build_case_statement(self, column: str, mapping: dict) -> str:
        cases = [f"WHEN {column} = '{value}' THEN {code}" for value, code in mapping.items()]
        return f"CASE {' '.join(cases)} ELSE NULL END AS {column}"

    def create_encoded_table(self, source_table: str = None, target_table: str = None):
        if source_table is None:
            source_table = f"{self.config.CATALOG}.{self.config.SCHEMA}.{self.config.TABLE_NAME}"
        if target_table is None:
            target_table = f"{self.config.CATALOG}.{self.config.SCHEMA}.{self.config.TABLE_BRONZE_NAME}"
        
        print("=" * 50)
        print("BRONZE ENCODING - CREATE ENCODED TABLE")
        print("=" * 50)

        select_parts = []

        for field in FIELDS_NOT_ENCODED:
            select_parts.append(field)

        print("\nmes mapping:")
        for value, code in MONTH_MAPPING.items():
            print(f"  {value} -> {code}")
        select_parts.append(self.build_case_statement('mes', MONTH_MAPPING))

        for field in FIELDS_TO_ENCODE:
            unique_values = self.get_unique_values(source_table, field)
            mapping = self.create_encoding_mapping(unique_values)
            print(f"\n{field} mapping:")
            for value, code in mapping.items():
                print(f"  {value} -> {code}")
            select_parts.append(self.build_case_statement(field, mapping))

        select_clause = ",\n    ".join(select_parts)
        query = f"""
            CREATE OR REPLACE TABLE {target_table}
            USING DELTA
            AS
            SELECT
                {select_clause}
            FROM {source_table}
        """

        print(f"\nCreating table {target_table}...")
        self.sql_executor.execute_sql_query(
            self.config.CATALOG,
            self.config.SCHEMA,
            query
        )
        print(f"  Table {target_table} created successfully")

    def validate_encoded_table(self, table_name: str) -> dict:
        query = f"SELECT COUNT(*) AS total_rows FROM {table_name}"
        print(f"\nValidating table {table_name}...")
        result = self.sql_executor.execute_sql_query(
            self.config.CATALOG,
            self.config.SCHEMA,
            query
        )
        print(f"  Validation result: {result['data']}")
        return result


def encode_bronze_table():
    encoder = BronzeEncoder()
    source_table = f"{encoder.config.CATALOG}.{encoder.config.SCHEMA}.{encoder.config.TABLE_NAME}"
    target_table = f"{encoder.config.CATALOG}.{encoder.config.SCHEMA}.{encoder.config.TABLE_BRONZE_NAME}"

    encoder.create_encoded_table(source_table, target_table)
    encoder.validate_encoded_table(target_table)

    print("=" * 50)
    print("BRONZE ENCODING COMPLETE")
    print("=" * 50)

