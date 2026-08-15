from typing import List, Dict, Any

class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k
        print(f"[Fusion] Initializing RRF Fusion (k={k})...")
        
    def fuse(self, milvus_results: List[Dict[str, Any]], es_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) for late fusion of Vector and Text results.
        Score = 1 / (k + rank_milvus) + 1 / (k + rank_es)
        """
        print("[Fusion] Executing RRF on RAM...")
        
        if not es_results:
            # If no ES text search was needed, just return Milvus top 100 with normalized scores
            for rank, hit in enumerate(milvus_results):
                hit["rrf_score"] = 1.0 / (self.k + rank + 1)
            return milvus_results
            
        # Build rank dictionaries
        milvus_ranks = {hit["shot_id"]: rank + 1 for rank, hit in enumerate(milvus_results)}
        es_ranks = {hit["shot_id"]: rank + 1 for rank, hit in enumerate(es_results)}
        
        # Collect all unique IDs
        all_ids = set(milvus_ranks.keys()).union(set(es_ranks.keys()))
        
        fused_hits = {}
        for shot_id in all_ids:
            score = 0.0
            if shot_id in milvus_ranks:
                score += 1.0 / (self.k + milvus_ranks[shot_id])
            if shot_id in es_ranks:
                score += 1.0 / (self.k + es_ranks[shot_id])
                
            # Merge original metadata (start_ms, end_ms, keyframe_id etc from Milvus/ES)
            hit_data = None
            for hit in milvus_results:
                if hit["shot_id"] == shot_id:
                    hit_data = hit.copy()
                    break
            
            if not hit_data:
                for hit in es_results:
                    if hit["shot_id"] == shot_id:
                        hit_data = hit.copy()
                        break
                        
            hit_data["rrf_score"] = score
            fused_hits[shot_id] = hit_data
            
        # Sort by RRF score descending
        sorted_hits = sorted(fused_hits.values(), key=lambda x: x["rrf_score"], reverse=True)
        print(f"[Fusion] RRF complete. Total candidates: {len(sorted_hits)}")
        
        return sorted_hits[:100] # Return Top 100
