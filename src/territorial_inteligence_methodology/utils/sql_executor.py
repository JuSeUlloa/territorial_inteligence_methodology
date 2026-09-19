from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient


class SqlExecutor:
    
    
    def __init__(self):
        self.client = DataBricksClient().get_client()
    
    def execute_sql_query(self, catalog_name: str, schema_name: str, query: str) -> dict:

        print(f'Executing SQL query: {query}')
        warehouse_id = DataBricksClient().get_warehouse_id()
        
        response = self.client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query,
            catalog=catalog_name,
            schema=schema_name,
            wait_timeout='30s'
        )
        
        state = response.status.state.value

        if state == 'SUCCEEDED':
            data = []
            if response.result and response.result.data_array:
                columns = [col.name for col in response.manifest.schema.columns]
                for row in response.result.data_array:
                    data.append(dict(zip(columns, row)))
            return {
                'status': state,
                'data': data,
                'statement_id': response.statement_id
            }
        else:
            error_info = response.status.error
            error_message = error_info.message if error_info else 'Unknown error'
            raise Exception(f'SQL execution failed ({state}): {error_message}')
