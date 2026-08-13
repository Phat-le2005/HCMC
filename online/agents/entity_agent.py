from typing import Dict, Any, List
from online.agents.base_agent import BaseAgent
from online.core.db_clients import db_manager

class EntityAgent(BaseAgent):
    def search(self, query: str, parsed_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        entities = parsed_json.get("entities", [])
        if not entities:
            return []
            
        print(f"[{self.name}] Searching for Entities/OCR: {entities}")
        
        # TODO: Query Elasticsearch (atsme_local_text) and Milvus (atsme_object_vectors)
        # For now, return a mock evidence packet
        
        evidence = self.format_evidence(
            candidate_id="mock_video_001",
            evidence_type="entities",
            evidence_data=[{"entity": e, "track_id": f"track_{i}"} for i, e in enumerate(entities)],
            score=0.78
        )
        return [evidence]
