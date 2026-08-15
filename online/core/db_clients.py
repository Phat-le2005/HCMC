from typing import Optional
from pymilvus import connections
from elasticsearch import Elasticsearch
from online.core.config import config

class DBClientManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBClientManager, cls).__new__(cls)
            cls._instance._init_clients()
        return cls._instance

    def _init_clients(self):
        self._es_client: Optional[Elasticsearch] = None
        self._milvus_connected: bool = False

    def get_elasticsearch(self) -> Elasticsearch:
        if self._es_client is None:
            print("[OnlineDB] Connecting to Elasticsearch...")
            self._es_client = Elasticsearch(config.elasticsearch_host)
        return self._es_client

    def connect_milvus(self):
        if not self._milvus_connected:
            print("[OnlineDB] Connecting to Milvus...")
            connections.connect("default", host=config.milvus_host, port=config.milvus_port)
            self._milvus_connected = True
            
    def close_all(self):
        if self._milvus_connected:
            connections.disconnect("default")
            self._milvus_connected = False
        if self._es_client:
            self._es_client.close()
            self._es_client = None

# Singleton instance
db_manager = DBClientManager()
