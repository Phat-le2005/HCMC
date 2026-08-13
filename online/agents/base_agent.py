from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgent(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def search(self, query: str, parsed_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute search based on the query and router's parsed JSON.
        Returns a list of evidence packets.
        """
        pass

    def format_evidence(self, candidate_id: str, evidence_type: str, evidence_data: List[Any], score: float) -> Dict[str, Any]:
        """
        Standardize the output format across all agents.
        """
        return {
            "candidate_id": candidate_id,
            "evidence": {
                evidence_type: evidence_data
            },
            "scores": {
                evidence_type: score
            }
        }
