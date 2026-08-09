import os
import json
import argparse
from pathlib import Path

# (Graph + DB logic)
# try:
#     from neo4j import GraphDatabase
#     from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
#     from elasticsearch import Elasticsearch
# except ImportError:
#     pass

from config_v5 import config

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def temporal_overlap(start1, end1, start2, end2) -> float:
    """Calculates temporal overlap in seconds between two segments."""
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

def build_graph(input_dir: Path, output_dir: Path):
    print(f"[Module 3] Loading data from {input_dir}")
    
    scenes_file = input_dir / "scenes.json"
    shots_file = input_dir / "shots.json"
    tracklets_file = input_dir / "tracklets.json"
    actions_file = input_dir / "actions.json"
    lexical_file = input_dir / "lexical_global.json"
    
    # Load data gracefully using safe loader
    scenes = load_json_safe(scenes_file) or []
    shots = load_json_safe(shots_file) or []
    
    tracklets_data = load_json_safe(tracklets_file) or {"tracklets": [], "static_objects": []}
    tracklets = tracklets_data.get("tracklets", [])
    static_objects = tracklets_data.get("static_objects", [])
    
    actions = load_json_safe(actions_file) or []
    lexical_global = load_json_safe(lexical_file) or {"shots": []}
    
    # Create temporal intersection index
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
                
    # Build a simulated JSON Graph structure
    graph_nodes = {
        "Scenes": len(scenes),
        "Shots": len(shots),
        "Tracklets": len(tracklets),
        "StaticObjects": len(static_objects),
        "Actions": len(actions),
        "Lexical": len(lexical_global.get("shots", []))
    }
    
    graph_edges = {
        "SCENE_HAS_SHOT": sum(len(s.get("shot_ids", [])) for s in scenes),
        "SHOT_HAS_TRACKLET": sum(len(ts) for ts in shot_to_tracklets.values()),
        "TRACKLET_HAS_ACTION": len(actions),
        "SHOT_HAS_LEXICAL": len([s for s in lexical_global.get("shots", []) if s.get("ocr") or s.get("asr")])
    }
    
    # Simulation: Write out the graph connections that would go into Neo4j
    graph_export = {
        "shot_to_tracklet_mapping": shot_to_tracklets,
        "nodes": graph_nodes,
        "edges": graph_edges
    }
    
    ensure_dir(output_dir)
    out_graph = output_dir / "graph_topology.json"
    with open(out_graph, "w", encoding="utf-8") as f:
        json.dump(graph_export, f, indent=2)
        
    print(f"[Module 3] Exported Neo4j graph topology to {out_graph}")
    print(f"[Module 3] Graph Summary: {graph_nodes} | {graph_edges}")
    
    # In a real environment, you would push to Neo4j, Milvus, and ElasticSearch here:
    # driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
    # connections.connect("default", host=config.milvus_host, port=config.milvus_port)
    # es = Elasticsearch(config.elasticsearch_host)
    print("[Module 3] Note: DB injection to Neo4j, Milvus, ElasticSearch is skipped in simulation.")

def main():
    parser = argparse.ArgumentParser(description="Module 3 – Graph Builder")
    parser.add_argument("--input-dir", type=str, default=config.intermediate_dir)
    parser.add_argument("--output-dir", type=str, default=config.output_dir)
    args = parser.parse_args()

    build_graph(Path(args.input_dir), Path(args.output_dir))

if __name__ == "__main__":
    main()
