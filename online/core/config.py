import os
from dataclasses import dataclass, field

@dataclass
class OnlineConfig:
    # -------------------- Service Endpoints --------------------
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "1459097e")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "ceFmdO3Rd2PmYaHMRmXrnlEEtze7DaMvNBgE-Kg3aqE")
    
    milvus_host: str = os.environ.get("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.environ.get("MILVUS_PORT", "19530"))
    
    elasticsearch_host: str = os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
    
    # -------------------- DB Collections/Indices --------------------
    milvus_scene_collection: str = "atsme_scene_vectors"
    milvus_shot_collection: str = "atsme_shot_vectors"
    milvus_object_collection: str = "atsme_object_vectors"
    milvus_event_collection: str = "atsme_event_vectors"
    
    es_global_index: str = "atsme_global_text"
    es_local_index: str = "atsme_local_text"

    # -------------------- Model Configs --------------------
    router_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    vision_language_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # -------------------- Search Thresholds --------------------
    top_k_spatial: int = 500
    top_k_entity: int = 500
    top_k_event: int = 500
    
    early_exit_threshold: float = 0.95
    hard_query_threshold: float = 0.80

config = OnlineConfig()
