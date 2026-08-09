from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def get_milvus_collections(host='localhost', port=19530):
    connections.connect("default", host=host, port=port)
    
    # 1. Scene Vectors (Dim: 1536 from fused SigLIP + WavLM)
    scene_fields = [
        FieldSchema(name="scene_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="start_ms", dtype=DataType.INT64),
        FieldSchema(name="end_ms", dtype=DataType.INT64),
        FieldSchema(name="news_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="fused_vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    scene_schema = CollectionSchema(fields=scene_fields, description="Scene Multimodal Vectors")
    
    # 2. Object Vectors (Dim: 768 from SigLIP)
    object_fields = [
        FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="shot_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="class_label", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="siglip_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    object_schema = CollectionSchema(fields=object_fields, description="Static Object Vectors")
    
    # 3. Action Vectors (Dim: 256 from mock InternVideo2)
    action_fields = [
        FieldSchema(name="track_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="action_label", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="action_vector", dtype=DataType.FLOAT_VECTOR, dim=256)
    ]
    action_schema = CollectionSchema(fields=action_fields, description="Action Vectors")
    
    collections = {}
    for name, schema in zip(["scene_vectors", "object_vectors", "action_vectors"], 
                          [scene_schema, object_schema, action_schema]):
        if not utility.has_collection(name):
            col = Collection(name=name, schema=schema)
            # Create HNSW index for vector field
            vec_field = [f.name for f in schema.fields if f.dtype == DataType.FLOAT_VECTOR][0]
            col.create_index(field_name=vec_field, index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}})
            collections[name] = col
        else:
            collections[name] = Collection(name)
            
    return collections
