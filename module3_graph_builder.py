import os
import json
import argparse
from pathlib import Path

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
        "actions": actions,
        "shot_to_tracklets": shot_to_tracklets
    }
    
    if dry_run:
        print("[Module 3] DRY-RUN MODE: Skipping actual DB ingestion.")
        # Simulated topology export
        graph_nodes = {
            "Scenes": len(scenes),
            "Shots": len(shots),
            "Tracklets": len(tracklets),
            "StaticObjects": len(static_objects),
            "Actions": len(actions)
        }
        ensure_dir(output_dir)
        out_graph = output_dir / "graph_topology_dryrun.json"
        with open(out_graph, "w", encoding="utf-8") as f:
            json.dump({"nodes": graph_nodes, "shot_to_tracklets": shot_to_tracklets}, f, indent=2)
        print(f"[Module 3] Exported Neo4j graph topology to {out_graph}")
        return

    print("[Module 3] Initializing DB Clients...")
    try:
        from neo4j import GraphDatabase
        from elasticsearch import Elasticsearch
        
        # 1. Neo4j Graph DB
        print("  -> Connecting to Neo4j...")
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        create_constraints(driver)
        ingest_graph(driver, db_data)
        driver.close()
        print("     [OK] Neo4j ingestion complete.")
        
        # 2. Milvus Vector DB
        print("  -> Connecting to Milvus...")
        collections = get_milvus_collections(host=config.milvus_host, port=config.milvus_port)
        
        # Insert Scenes (Fused Vector)
        scene_data = [[], [], [], [], [], []]
        for sc in scenes:
            scene_data[0].append(sc["scene_id"])
            scene_data[1].append(metadata.get("video_id", ""))
            scene_data[2].append(sc["start_ms"])
            scene_data[3].append(sc["end_ms"])
            scene_data[4].append(sc.get("news_type", "unknown"))
            # Need to get fused vector from shots
            fused_vec = [0.0]*1536
            # Basic fallback: find first shot in scene
            for sh_id in sc.get("shot_ids", []):
                for sh in shots:
                    if sh["shot_id"] == sh_id:
                        img_vec = sh.get("image_vector", [0.0]*768)
                        aud_vec = sh.get("audio_vector", [0.0]*768)
                        if len(img_vec) == 768 and len(aud_vec) == 768:
                            fused_vec = img_vec + aud_vec
                        break
                break
            scene_data[5].append(fused_vec)
        if scene_data[0]:
            collections["scene_vectors"].insert(scene_data)
            
        # Insert Static Objects (SigLIP Vector)
        obj_data = [[], [], [], [], []]
        for so in static_objects:
            obj_data[0].append(so["object_id"])
            obj_data[1].append(so.get("shot_id", ""))
            obj_data[2].append(metadata.get("video_id", ""))
            obj_data[3].append(so.get("class_label", ""))
            vec = so.get("siglip_vector", [])
            if not vec or len(vec) != 768:
                vec = [0.0] * 768
            obj_data[4].append(vec)
        if obj_data[0]:
            collections["object_vectors"].insert(obj_data)
            
        # Insert Actions (Action Vector)
        act_data = [[], [], [], []]
        for ac in actions:
            act_data[0].append(ac["track_id"])
            act_data[1].append(metadata.get("video_id", ""))
            act_data[2].append(ac.get("action_label", ""))
            vec = ac.get("action_vector", [])
            if not vec or len(vec) != 256:
                vec = [0.0] * 256
            act_data[3].append(vec)
        if act_data[0]:
            collections["action_vectors"].insert(act_data)
            
        for col in collections.values():
            col.flush()
        print("     [OK] Milvus ingestion complete.")

        # 3. Elasticsearch
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
