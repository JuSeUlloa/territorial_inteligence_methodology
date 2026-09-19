from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.client_api import Client_Api


def load_holidays():
    config = Config()
    api_client = Client_Api()
    
    print("=" * 50)
    print("HOLIDAYS API LOADER")
    print("=" * 50)
    
    
    api_client.create_schema(config.CATALOG, config.SCHEMA)
    api_client.create_holidays_table(config.CATALOG, config.SCHEMA)
    
    
    for country in config.HOLIDAYS_COUNTRIES:
        for year in config.HOLIDAYS_YEARS:
            print(f"\nProcessing {country} - {year}...")
            
            # Construir URL
            url = config.HOLIDAYS_API_URL.format(pais=country, year=year)
            print(f"URL: {url}")
            
            data = api_client.fetch_data(url,30)
            print(f"  Fetched {len(data)} holidays")
            
            
            api_client.save_json_as_delta(
                data=data,
                catalog=config.CATALOG,
                schema=config.SCHEMA,
                table='holidays'
            )
    
    api_client.validate_table(config.CATALOG, config.SCHEMA, 'holidays')
    
    print("=" * 50)
    print("HOLIDAYS LOAD COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    load_holidays()