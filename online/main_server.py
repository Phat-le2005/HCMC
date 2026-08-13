import sys
import os
import json
import asyncio
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from online.core.db_clients import db_manager
from online.router.qwen_router import QwenRouter
from online.agents.spatial_agent import SpatialAgent
from online.agents.entity_agent import EntityAgent
from online.agents.event_agent import EventAgent
from online.fusion.mlp_scorer import MLPScorer

async def run_query(query_text: str):
    print(f"\n{'='*50}\n[Main] Processing Query: '{query_text}'\n{'='*50}")
    
    # 1. Routing
    router = QwenRouter()
    parsed_json = router.parse_query(query_text)
    
    # 2. Agent Swarm (Fan-out)
    spatial_agent = SpatialAgent()
    entity_agent = EntityAgent()
    event_agent = EventAgent()
    
    print("\n[Main] Dispatching to Agent Swarm...")
    # In a real async environment, use asyncio.gather for parallel execution
    spatial_evidence = spatial_agent.search(query_text, parsed_json)
    entity_evidence = entity_agent.search(query_text, parsed_json)
    event_evidence = event_agent.search(query_text, parsed_json)
    
    all_evidence = spatial_evidence + entity_evidence + event_evidence
    
    # 3. Fusion
    print("\n[Main] Merging and Fusing Scores...")
    fusion_module = MLPScorer()
    final_results = fusion_module.fuse_scores(all_evidence)
    
    print(f"\n[Main] Top Results:")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))
    
    # 4. Latency Optimizer (Early Exit Check)
    if final_results and final_results[0]["confidence_score"] >= 0.95:
        print("\n[Main] SUCCESS: Adaptive Early Exit Triggered! (Score >= 0.95)")
    else:
        print("\n[Main] HARD QUERY: Proceeding to Phase 3 (Cross-Attention & VLM Judge)...")

if __name__ == "__main__":
    test_query = "Tìm chiếc xe buýt màu xanh đang chạy nhanh trên đường phố ban ngày"
    asyncio.run(run_query(test_query))
    
    db_manager.close_all()
