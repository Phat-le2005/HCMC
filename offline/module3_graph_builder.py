import os
import json
import argparse
from pathlib import Path
import numpy as np

from config_v5 import config
from schemas.milvus_schema import get_milvus_collections
from schemas.neo4j_schema import create_constraints, ingest_graph
from schemas.elasticsearch_schema import create_indices, bulk_index

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def temporal_overlap(start1, end1, start2, end2) -> float:
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start) / 1000.0

def load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def validate_record(record: dict, schema: dict, record_name: str) -> bool:
    """Validate a single record against expected field types and dimensions."""
    valid = True
    for field_name, spec in schema.items():
        value = record.get(field_name)
        
        # Check required fields
        if spec.get("required") and value is None:
            print(f"  [WARN] {record_name}: Missing required field '{field_name}'")
            valid = False
            continue
        
        if value is None:
            continue
            
        # Check type
        expected_type = spec.get("type")
        if expected_type and not isinstance(value, expected_type):
            print(f"  [WARN] {record_name}: Field '{field_name}' expected {expected_type.__name__}, got {type(value).__name__}")
            valid = False
            
        # Check vector dimension
        expected_dim = spec.get("dim")
        if expected_dim and isinstance(value, list) and len(value) != expected_dim:
            print(f"  [WARN] {record_name}: Vector '{field_name}' expected dim={expected_dim}, got dim={len(value)}")
            valid = False
            
        # Check max string length (for Milvus VARCHAR)
        max_len = spec.get("max_len")
        if max_len and isinstance(value, str) and len(value) > max_len:
            print(f"  [WARN] {record_name}: String '{field_name}' exceeds max_len={max_len} (len={len(value)}), truncating")
            record[field_name] = value[:max_len]
            
    return valid

def validate_all_data(db_data: dict) -> dict:
    """Pre-validate all data before DB insertion. Returns sanitized data with stats."""
    
    SCENE_SCHEMA = {
        "scene_id": {"type": str, "required": True, "max_len": 100},
        "start_ms": {"type": (int, float), "required": True},
        "end_ms": {"type": (int, float), "required": True},
        "news_type": {"type": str, "max_len": 50},
    }
    SHOT_SCHEMA = {
        "shot_id": {"type": str, "required": True, "max_len": 100},
        "start_ms": {"type": (int, float), "required": True},
        "global_ocr": {"type": str},
        "global_asr": {"type": str},
        "image_vector": {"type": list, "dim": 768},
        "audio_vector": {"type": list, "dim": 768},
    }
    TRACKLET_SCHEMA = {
        "track_id": {"type": str, "required": True, "max_len": 100},
        "class_label": {"type": str, "required": True, "max_len": 100},
        "start_ms": {"type": (int, float), "required": True},
        "end_ms": {"type": (int, float), "required": True},
    }
    STATIC_OBJ_SCHEMA = {
        "object_id": {"type": str, "required": True, "max_len": 100},
        "shot_id": {"type": str, "max_len": 100},
        "class_label": {"type": str, "required": True, "max_len": 100},
        "siglip_vector": {"type": list, "dim": 768},
    }
    EVENT_SCHEMA = {
        "track_id": {"type": str, "required": True, "max_len": 100},
        "action_label": {"type": str, "required": True, "max_len": 100},
        "action_vector": {"type": list, "dim": 256},
    }

    stats = {"scenes_valid": 0, "scenes_invalid": 0, "shots_valid": 0, "shots_invalid": 0,
             "tracklets_valid": 0, "tracklets_invalid": 0, "static_valid": 0, "static_invalid": 0,
             "events_valid": 0, "events_invalid": 0}
    
    print("[Module 3] Validating data before DB insertion...")
    
    for sc in db_data.get("scenes", []):
        if validate_record(sc, SCENE_SCHEMA, f"Scene[{sc.get('scene_id', '?')}]"):
            stats["scenes_valid"] += 1
        else:
            stats["scenes_invalid"] += 1
    
    for sh in db_data.get("shots", []):
        if validate_record(sh, SHOT_SCHEMA, f"Shot[{sh.get('shot_id', '?')}]"):
            stats["shots_valid"] += 1
        else:
            stats["shots_invalid"] += 1
    
    for tr in db_data.get("tracklets", []):
        if validate_record(tr, TRACKLET_SCHEMA, f"Tracklet[{tr.get('track_id', '?')}]"):
            stats["tracklets_valid"] += 1
        else:
            stats["tracklets_invalid"] += 1
    
    for so in db_data.get("static_objects", []):
        if validate_record(so, STATIC_OBJ_SCHEMA, f"StaticObj[{so.get('object_id', '?')}]"):
            stats["static_valid"] += 1
        else:
            stats["static_invalid"] += 1
    
    for ac in db_data.get("actions", []):
        if validate_record(ac, EVENT_SCHEMA, f"Event[{ac.get('track_id', '?')}]"):
            stats["events_valid"] += 1
        else:
            stats["events_invalid"] += 1
    
    total_invalid = stats["scenes_invalid"] + stats["shots_invalid"] + stats["tracklets_invalid"] + stats["static_invalid"] + stats["events_invalid"]
    print(f"[Module 3] Validation complete: {total_invalid} issue(s) found.")
    for k, v in stats.items():
        if v > 0:
            print(f"    {k}: {v}")
    
    return stats

def build_graph(input_dir: Path, output_dir: Path, dry_run: bool = False):
    print(f"[Module 3] Loading data from {input_dir}")
    
    scenes_file = input_dir / "scenes.json"
    shots_file = input_dir / "shots.json"
    tracklets_file = input_dir / "tracklets.json"
    actions_file = input_dir / "actions.json"
    lexical_file = input_dir / "lexical_global.json"
    
    scenes = load_json_safe(scenes_file) or []
    shots = load_json_safe(shots_file) or []
    
    tracklets_data = load_json_safe(tracklets_file) or {"tracklets": [], "static_objects": [], "metadata": {}}
    tracklets = tracklets_data.get("tracklets", [])
    static_objects = tracklets_data.get("static_objects", [])
    ocr_local = tracklets_data.get("ocr_local", [])
    metadata = tracklets_data.get("metadata", {})
    
    actions = load_json_safe(actions_file) or []
    lexical_global = load_json_safe(lexical_file) or {"shots": []}
    
    print(f"[Module 3] Running temporal intersection matching (threshold: {config.overlap_threshold}s)...")
    shot_to_tracklets = {s["shot_id"]: [] for s in shots}
    
    for shot in shots:
        s_start = shot["start_ms"]
        s_end = shot["end_ms"]
        for track in tracklets:
            t_start = track["start_ms"]
            t_end = track["end_ms"]
            overlap = temporal_overlap(s_start, s_end, t_start, t_end)
            if overlap > config.overlap_threshold:
                shot_to_tracklets[shot["shot_id"]].append(track["track_id"])
                
    # Data aggregation object for graph injection
    db_data = {
        "video_metadata": metadata,
        "scenes": scenes,
        "shots": shots,
        "tracklets": tracklets,
        "static_objects": static_objects,
        "ocr_local": ocr_local,
        "actions": actions,
        "shot_to_tracklets": shot_to_tracklets
    }
    
    # Always validate data format before any output
    validation_stats = validate_all_data(db_data)
    
    if dry_run:
        print("[Module 3] DRY-RUN MODE: Skipping actual DB ingestion.")
        # Simulated topology export
        graph_nodes = {
            "Scenes": len(scenes),
            "Shots": len(shots),
            "Tracklets": len(tracklets),
            "StaticObjects": len(static_objects),
            "Events": len(actions)
        }
        
        topology_export = {
            "node_counts": graph_nodes,
            "shot_to_tracklets": shot_to_tracklets,
            "sample_records": {
                "scene": scenes[0] if scenes else None,
                "shot": shots[0] if shots else None,
                "tracklet": tracklets[0] if tracklets else None,
                "static_object": static_objects[0] if static_objects else None,
                "event": actions[0] if actions else None
            }
        }
        
        ensure_dir(output_dir)
        out_graph = output_dir / "graph_topology_dryrun.json"
        with open(out_graph, "w", encoding="utf-8") as f:
            json.dump(topology_export, f, indent=2)
        print(f"[Module 3] Exported Neo4j graph topology to {out_graph}")
        return

    print("[Module 3] Initializing DB Clients...")
    try:
        from elasticsearch import Elasticsearch
        import json
        
        # Extract FPS for keyframe_id calculation
        original_fps = lexical_global.get("original_fps", 25.0)
        
        # 1. Milvus Vector DB (Flattened Architecture)
        print("  -> Connecting to Milvus...")
        collections = get_milvus_collections(host=config.milvus_host, port=config.milvus_port)
        
        shot_data = [[], [], [], [], []]
        for sh in shots:
            shot_id = sh["shot_id"]
            start_ms = sh["start_ms"]
            end_ms = sh["end_ms"]
            
            # Calculate keyframe_id (middle frame)
            keyframe_id = int(((start_ms + end_ms) / 2000.0) * original_fps)
            
            # Build payload (Flattened Graph)
            entities = []
            
            # Add Static Objects associated with this shot
            for so in static_objects:
                if so.get("shot_id") == shot_id:
                    entities.append(f"object_{so.get('class_label', '')}")
            
            # Add OCR
            for ocr in ocr_local:
                if ocr.get("shot_id") == shot_id:
                    entities.append(f"ocr_{ocr.get('text', '')}")
                    
            # Add Events (Tracklets overlapping with this shot)
            for track_id in shot_to_tracklets.get(shot_id, []):
                for ac in actions:
                    if ac["track_id"] == track_id:
                        entities.append(f"event_{ac.get('action_label', '')}")
            
            # Include global OCR/ASR
            for lex_sh in lexical_global.get("shots", []):
                if lex_sh["shot_id"] == shot_id:
                    if lex_sh.get("ocr"): entities.append(f"ocr_{lex_sh['ocr']}")
                    if lex_sh.get("asr"): entities.append(f"asr_{lex_sh['asr']}")
            
            payload_str = json.dumps(entities, ensure_ascii=False)
            
            img_vec = sh.get("image_vector", [0.0]*768)
            aud_vec = sh.get("audio_vector", [0.0]*768)
            fused_vec = img_vec + aud_vec if len(img_vec) == 768 and len(aud_vec) == 768 else [0.0]*1536
            
            shot_data[0].append(shot_id)
            shot_data[1].append(metadata.get("video_id", lexical_global.get("video_id", "")))
            shot_data[2].append(keyframe_id)
            shot_data[3].append(payload_str)
            shot_data[4].append(fused_vec)

        if shot_data[0]:
            collections[config.milvus_shot_collection].insert(shot_data)
            collections[config.milvus_shot_collection].flush()
        print("     [OK] Milvus flattened ingestion complete.")

        # 2. Elasticsearch
        print("  -> Connecting to Elasticsearch...")
        es = Elasticsearch(config.elasticsearch_host)
        create_indices(es)
        bulk_index(es, db_data)
        print("     [OK] Elasticsearch indexing complete.")

    except Exception as e:
        print(f"[ERROR] DB ingestion failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Module 3 – Graph Builder")
    parser.add_argument("--input-dir", type=str, default=config.intermediate_dir)
    parser.add_argument("--output-dir", type=str, default=config.output_dir)
    parser.add_argument("--dry-run", action="store_true", help="Run topology matching without pushing to DBs")
    args = parser.parse_args()

    build_graph(Path(args.input_dir), Path(args.output_dir), args.dry_run)

if __name__ == "__main__":
    main()
