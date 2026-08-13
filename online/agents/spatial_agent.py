from typing import Dict, Any, List
from online.agents.base_agent import BaseAgent
from online.core.db_clients import db_manager

class SpatialAgent(BaseAgent):
    def search(self, query: str, parsed_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        scene_query = parsed_json.get("scene", "")
        if not scene_query:
            return []
            
        print(f"[{self.name}] Searching for Scene/Context: {scene_query}")
        
        # TODO: Encode scene_query to SigLIP vector and query Milvus
        # For now, return a mock evidence packet
        
        evidence = self.format_evidence(
            candidate_id="mock_video_001",
            evidence_type="scene",
            evidence_data=[{"matched_text": scene_query, "shot_id": "shot_001"}],
            score=0.85
        )
        return [evidence]
