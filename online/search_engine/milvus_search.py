from typing import List, Dict, Any
from online.core.config import config
from online.core.db_clients import db_manager
from pymilvus import Collection

class MilvusSearcher:
    def __init__(self):
        db_manager.connect_milvus()
        try:
            self.shot_collection = Collection(config.milvus_shot_collection)
            self.shot_collection.load()
            print(f"[MilvusSearcher] Loaded collection {config.milvus_shot_collection}")
        except Exception as e:
            print(f"[MilvusSearcher] Warning: Could not load collection: {e}")

    def search_shots(self, query_vector: List[float], filter_entities: List[str] = None, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Executes a vector search on Milvus with scalar payload filtering.
        """
        print("[MilvusSearcher] Executing Vector Search with Payload Filter...")
        
        # Build Milvus scalar filter expression based on entities
        expr = None
        if filter_entities:
            # e.g., filter out shots where 'entities_payload' contains specific words
            # In Milvus 2.x, array_contains or JSON payload matches can be used.
            # Example expr: "entities_payload LIKE '%xe buýt%'"
            expr_parts = [f"entities_payload LIKE '%{e}%'" for e in filter_entities]
            expr = " AND ".join(expr_parts)
            print(f"                 Filter Expr: {expr}")
            
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        try:
            results = self.shot_collection.search(
                data=[query_vector],
                anns_field="fused_vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["shot_id", "video_id", "start_ms", "end_ms"]
            )
            
            hits = []
            for hit in results[0]:
                hits.append({
                    "shot_id": hit.entity.get("shot_id"),
                    "video_id": hit.entity.get("video_id"),
                    "start_ms": hit.entity.get("start_ms"),
                    "end_ms": hit.entity.get("end_ms"),
                    "score": hit.distance
                })
            return hits
        except Exception as e:
            print(f"[MilvusSearcher] Error during search: {e}")
            # Mock return for pipeline testing if DB is down
            return [{"shot_id": f"mock_shot_{i}", "score": 0.9} for i in range(top_k)]
