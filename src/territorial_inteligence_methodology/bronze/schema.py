from territorial_inteligence_methodology.utils.domains import COLUMNS_MAPPING


class SchemaGenerator:
    
    
    def __init__(self):
        self.columns_mapping = COLUMNS_MAPPING
    
    def build_select_query(self, source: str) -> str:
       
        select_columns = ', '.join(
            [f"`{original}` AS {mapped}" for original, mapped in self.columns_mapping.items()]
        )
        query = f"SELECT {select_columns} FROM {source}"
        return query
    
    def get_columns(self) -> list:
        return list(self.columns_mapping.values())
    
    def get_original_columns(self) -> list:
        return list(self.columns_mapping.keys())
    
    def validate_columns(self, available_columns: list) -> bool:
        missing = set(self.get_original_columns()) - set(available_columns)
        if missing:
            raise ValueError(f'Missing columns in source: {missing}')
        return True
