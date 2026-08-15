from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from config_v5 import config

def get_milvus_collections(host='localhost', port=19530):
    connections.connect("default", host=host, port=port)
    
    # Shot Vectors (Dim: 1536 from fused SigLIP + WavLM) - FLATTENED SURVIVAL MODE
    shot_fields = [
        FieldSchema(name="shot_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="keyframe_id", dtype=DataType.INT64), # For TRAKE submission
        FieldSchema(name="entities_payload", dtype=DataType.VARCHAR, max_length=65535), # Contains all OCR, Objects, Events
        FieldSchema(name="fused_vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    shot_schema = CollectionSchema(fields=shot_fields, description="Flattened Shot Vectors for AIC")
    
    col_names = [config.milvus_shot_collection]
    schemas = [shot_schema]
    
    collections = {}
    for name, schema in zip(col_names, schemas):
        if utility.has_collection(name):
            utility.drop_collection(name)
        col = Collection(name, schema)
        
        # Create Index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        
        vec_field = "fused_vector"
        
        col.create_index(field_name=vec_field, index_params=index_params)
        col.load()
        collections[name] = col
        print(f"Created & Loaded Milvus Collection: {name}")
        
    return collections
