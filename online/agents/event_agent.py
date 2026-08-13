from typing import Dict, Any, List
from online.agents.base_agent import BaseAgent
from online.core.db_clients import db_manager

class EventAgent(BaseAgent):
    def search(self, query: str, parsed_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        events = parsed_json.get("events", [])
        if not events:
            return []
            
        print(f"[{self.name}] Searching for Actions/Events: {events}")
        
        # TODO: Encode events to InternVideo2/HGT vector and query Milvus
        # For now, return a mock evidence packet
        
        evidence = self.format_evidence(
            candidate_id="mock_video_001",
            evidence_type="events",
            evidence_data=[{"action": e, "event_id": f"evt_{i}"} for i, e in enumerate(events)],
            score=0.92
        )
        return [evidence]
