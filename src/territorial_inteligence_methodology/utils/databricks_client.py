from databricks.sdk import WorkspaceClient


class DataBricksClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._warehouse_id = None
        return cls._instance
    
    def get_client(self) -> WorkspaceClient:
        """Retorna la instancia singleton de WorkspaceClient."""
        if self._client is None:
            self._client = WorkspaceClient()
        return self._client
    
    def get_warehouse_id(self) -> str:
        """Obtiene el primer warehouse disponible y lo inicia si está detenido."""
        if self._warehouse_id is not None:
            return self._warehouse_id
        
        client = self.get_client()
        warehouses = list(client.warehouses.list())
        
        if not warehouses:
            raise Exception(
                "No warehouses found in the workspace. "
                "Create one in the Databricks UI."
            )
        
        warehouse_id = str(warehouses[0].id)
        self.start_warehouse(client, warehouse_id)
        self._warehouse_id = warehouse_id
        return warehouse_id
    
    def start_warehouse(self, client: WorkspaceClient, warehouse_id: str) -> str:
        warehouse = client.warehouses.get(id=warehouse_id)
        
        if warehouse.state.value == 'STOPPED':
            print(f'Starting warehouse {warehouse_id}...')
            client.warehouses.start_and_wait(id=warehouse_id)
            print(f'  Warehouse started')
        elif warehouse.state.value == 'RUNNING':
            print(f'Warehouse {warehouse_id} is already running')
        
        return warehouse_id
    
    def reset_warehouse_cache(self):
        self._warehouse_id = None
