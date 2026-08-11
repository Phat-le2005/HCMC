from elasticsearch import Elasticsearch, helpers
from config_v5 import config

def create_indices(es: Elasticsearch):
    # Global text index (OCR + ASR per shot)
    global_mapping = {
        "mappings": {
            "properties": {
                "shot_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "scene_id": {"type": "keyword"},
                "ocr_text": {"type": "text"},
                "ocr_text_raw": {"type": "text"},
                "asr_text": {"type": "text"},
                "news_type": {"type": "keyword"},
                "start_ms": {"type": "long"},
                "end_ms": {"type": "long"},
                "keyframe_path": {"type": "keyword"}
            }
        }
    }
    
    # Local text index (OCR per static object)
    local_mapping = {
        "mappings": {
            "properties": {
                "object_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "shot_id": {"type": "keyword"},
                "class_label": {"type": "keyword"},
                "ocr_text": {"type": "text"},
                "bbox": {"type": "float"}
            }
        }
    }
    
    if not es.indices.exists(index=config.es_global_index):
        es.indices.create(index=config.es_global_index, body=global_mapping)
        
    if not es.indices.exists(index=config.es_local_index):
        es.indices.create(index=config.es_local_index, body=local_mapping)

def bulk_index(es: Elasticsearch, data):
    actions = []
    video_id = data.get("video_metadata", {}).get("video_id", "unknown")
    
    # Create shot to scene mapping to retrieve scene_id for global text
    shot_to_scene = {}
    for sc in data.get("scenes", []):
        for sh_id in sc.get("shot_ids", []):
            shot_to_scene[sh_id] = sc["scene_id"]
            
    # Global Index
    for sh in data.get("shots", []):
        if sh.get("global_ocr") or sh.get("global_asr"):
            actions.append({
                "_index": config.es_global_index,
                "_id": sh["shot_id"],
                "_source": {
                    "shot_id": sh["shot_id"],
                    "video_id": video_id,
                    "scene_id": shot_to_scene.get(sh["shot_id"], "unknown"),
                    "ocr_text": sh.get("global_ocr", ""),
                    "ocr_text_raw": sh.get("global_ocr_raw", ""),
                    "asr_text": sh.get("global_asr", ""),
                    "news_type": sh.get("news_type", "unknown"),
                    "start_ms": sh.get("start_ms", 0),
                    "end_ms": sh.get("end_ms", 0),
                    "keyframe_path": sh.get("image_vector_path", "")
                }
            })
            
    # Local Index
    for so in data.get("static_objects", []):
        if so.get("ocr_text"):
            actions.append({
                "_index": config.es_local_index,
                "_id": so["object_id"],
                "_source": {
                    "object_id": so["object_id"],
                    "video_id": video_id,
                    "shot_id": so.get("shot_id", "unknown"),
                    "class_label": so.get("class_label", "unknown"),
                    "ocr_text": so.get("ocr_text", ""),
                    "bbox": so.get("bbox", [])
                }
            })
            
    if actions:
        helpers.bulk(es, actions)
