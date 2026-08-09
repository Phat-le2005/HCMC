from elasticsearch import Elasticsearch, helpers

def create_indices(es: Elasticsearch):
    # Global text index (OCR + ASR per shot)
    global_mapping = {
        "mappings": {
            "properties": {
                "shot_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "ocr_text": {"type": "text"},
                "asr_text": {"type": "text"},
                "news_type": {"type": "keyword"}
            }
        }
    }
    
    # Local text index (OCR per static object)
    local_mapping = {
        "mappings": {
            "properties": {
                "object_id": {"type": "keyword"},
                "shot_id": {"type": "keyword"},
                "class_label": {"type": "keyword"},
                "ocr_text": {"type": "text"}
            }
        }
    }
    
    if not es.indices.exists(index="atsme_global"):
        es.indices.create(index="atsme_global", body=global_mapping)
        
    if not es.indices.exists(index="atsme_local"):
        es.indices.create(index="atsme_local", body=local_mapping)

def bulk_index(es: Elasticsearch, data):
    actions = []
    
    # Global Index
    for sh in data.get("shots", []):
        if sh.get("global_ocr") or sh.get("global_asr"):
            actions.append({
                "_index": "atsme_global",
                "_id": sh["shot_id"],
                "_source": {
                    "shot_id": sh["shot_id"],
                    "video_id": data.get("video_metadata", {}).get("video_id", "unknown"),
                    "ocr_text": sh.get("global_ocr", ""),
                    "asr_text": sh.get("global_asr", ""),
                    "news_type": sh.get("news_type", "unknown")
                }
            })
            
    # Local Index
    for so in data.get("static_objects", []):
        if so.get("ocr_text"):
            actions.append({
                "_index": "atsme_local",
                "_id": so["object_id"],
                "_source": {
                    "object_id": so["object_id"],
                    "shot_id": so.get("shot_id", "unknown"),
                    "class_label": so.get("class_label", "unknown"),
                    "ocr_text": so.get("ocr_text", "")
                }
            })
            
    if actions:
        helpers.bulk(es, actions)
