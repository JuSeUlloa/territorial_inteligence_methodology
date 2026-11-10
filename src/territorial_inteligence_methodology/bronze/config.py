import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    
    
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent.parent
        env_path = project_root / '.env'
        
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        self.CATALOG = self._get_env('CATALOG', 'workspace')
        self.SCHEMA = self._get_env('SCHEMA', 'territorial_inteligence')
        self.VOLUME_NAME = self._get_env('VOLUME_NAME', 'hurto_personas')
        self.TABLE_NAME = self._get_env('TABLE_NAME', 'hurtos_personas')
        self.TABLE_BRONZE_NAME = self._get_env('BRONZE_TABLE_NAME', 'hurtos_personas_bronce')
        self.TABLE_SILVER_NAME = self._get_env('SILVER_TABLE_NAME', 'hurtos_personas_silver')
        self.FILE_NAME = self._get_env('FILE_NAME', 'dataset_hurto_personas.parquet')
        self.PREDICTION_TABLE_NAME = self._get_env('PREDICTION_TABLE_NAME', 'hurto_personas_prediccion')
        self.CONSOLIDATED_TABLE_NAME = self._get_env('CONSOLIDATED_TABLE_NAME', 'consolidado')
        self.USE_THRESHOLDS_SHRINKAGE = self._get_env('USE_THRESHOLDS_SHRINKAGE', 'False').lower() == 'true'
        self.TABLE_BRONZE_MUNICIPALITY_NAME = self._get_env('TABLE_BRONZE_MUNICIPALITY_NAME', 'hurto_personas_bronce_municipio')
        self.TABLE_SILVER_MUNICIPALITY_NAME = self._get_env('TABLE_SILVER_MUNICIPALITY_NAME', 'hurto_personas_silver_municipio')
        self.POPULATION_MUNICIPALITY_NAME = self._get_env('POPULATION_MUNICIPALITY_NAME', 'poblacion_municipio')
        self.MUNICIPIOS_BARRIOS = self._get_env('MUNICIPIOS_BARRIOS', '76109').split(',')
        
        self.HOLIDAYS_API_URL = self._get_env(
            'HOLIDAYS_API_URL',
            'https://calendariosnacionales.com/{pais}/v1/{year}/nacionales.json'
        )
        self.HOLIDAYS_COUNTRIES = self._get_env('HOLIDAYS_COUNTRIES', 'co').split(',')
        self.HOLIDAYS_YEARS = self._get_env('HOLIDAYS_YEARS', '2025,2026').split(',')
    
    def _get_env(self, key: str, default: str) -> str:
        value = os.getenv(key, default)
        if not value:
            raise ValueError(f'Config variable {key} is empty')
        return value
    
    @property
    def table_full_name(self) -> str:
        return f'{self.CATALOG}.{self.SCHEMA}.{self.TABLE_BRONZE_NAME}'
    
    @property
    def volume_path(self) -> str:
        return f'/Volumes/{self.CATALOG}/{self.SCHEMA}/{self.VOLUME_NAME}/files'

