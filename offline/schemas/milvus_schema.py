from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from config_v5 import config

def get_milvus_collections(host='localhost', port=19530):
    connections.connect("default", host=host, port=port)
    
    # 1. Scene Vectors (Dim: 1536 from fused SigLIP + WavLM)
    scene_fields = [
        FieldSchema(name="scene_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="start_ms", dtype=DataType.INT64),
        FieldSchema(name="end_ms", dtype=DataType.INT64),
        FieldSchema(name="news_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="keyframe_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="fused_vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    scene_schema = CollectionSchema(fields=scene_fields, description="Scene Multimodal Vectors")
    
    # 2. Shot Vectors (Dim: 1536 from fused SigLIP + WavLM)
    shot_fields = [
        FieldSchema(name="shot_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="start_ms", dtype=DataType.INT64),
        FieldSchema(name="end_ms", dtype=DataType.INT64),
        FieldSchema(name="news_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="keyframe_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="fused_vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    shot_schema = CollectionSchema(fields=shot_fields, description="Shot Multimodal Vectors")
    
    # 3. Object Vectors (Dim: 768 from SigLIP)
    object_fields = [
        FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="shot_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="class_label", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="siglip_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    object_schema = CollectionSchema(fields=object_fields, description="Static Object Vectors")
    
    # 4. Event Vectors (Dim: 256 from InternVideo2 / Mock)
    event_fields = [
        FieldSchema(name="event_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="track_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="action_label", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="action_vector", dtype=DataType.FLOAT_VECTOR, dim=256)
    ]
    event_schema = CollectionSchema(fields=event_fields, description="Action/Event Vectors")
    
    col_names = [
        config.milvus_scene_collection, 
        config.milvus_shot_collection,
        config.milvus_object_collection, 
        config.milvus_event_collection
    ]
    schemas = [scene_schema, shot_schema, object_schema, event_schema]
    
    collections = {}
    for name, schema in zip(col_names, schemas):
        if not utility.has_collection(name):
            col = Collection(name=name, schema=schema)
            # Create HNSW index for vector field
            vec_field = next(f.name for f in schema.fields if f.dtype == DataType.FLOAT_VECTOR)
            col.create_index(
                field_name=vec_field, 
                index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
            )
            collections[name] = col
        else:
            collections[name] = Collection(name)
            
    return collections
