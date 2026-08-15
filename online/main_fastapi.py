import time
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from online.router.regex_router import RegexRouter
from online.search_engine.milvus_search import MilvusSearcher
from online.search_engine.rrf_fusion import RRFFusion
from online.core.db_clients import db_manager

app = FastAPI(title="ATSME v6.0 - Survival Mode")

# Mount static and templates
app.mount("/static", StaticFiles(directory="online/web_ui/static"), name="static")
templates = Jinja2Templates(directory="online/web_ui/templates")

# Initialize Singletons
router = RegexRouter()
milvus_searcher = MilvusSearcher()
rrf_fusion = RRFFusion(k=60)

@app.on_event("shutdown")
def shutdown_event():
    db_manager.close_all()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": []})

@app.get("/search")
async def search_api(q: str):
    start_time = time.time()
    
    # 1. Light-speed Routing
    parsed_query = router.parse_query(q)
    
    # 2. Vector Search (Mock Vector for now)
    # In real pipeline: query_vector = clip_model.encode(q)
    mock_vector = [0.1] * 1536 
    
    # Milvus + Payload Filter
    milvus_hits = milvus_searcher.search_shots(
        query_vector=mock_vector,
        filter_entities=parsed_query.get("filter_entities", [])
    )
    
    # 3. Text Search (ES)
    es_hits = None
    if parsed_query.get("filter_ocr"):
        print("[ES] Searching exact text...")
        # Mock ES hits for testing
        es_hits = [{"shot_id": "mock_shot_0", "score": 10.5}, {"shot_id": "mock_shot_2", "score": 8.2}]
        
    # 4. RRF Fusion
    final_hits = rrf_fusion.fuse(milvus_hits, es_hits)
    
    latency = round((time.time() - start_time) * 1000, 2)

    print(f"--- Query resolved in {latency}ms ---")
    
    return {
        "status": "success",
        "latency_ms": latency,
        "hits": final_hits
    }
