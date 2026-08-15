import re
from typing import Dict, Any

class RegexRouter:
    def __init__(self):
        print("[Router] Initializing Regex Rule-based Router (Survival Mode)...")
        # Chỉ bắt các keyword đặc biệt để filter, phần còn lại nhường cho SigLIP Text Encoder
        self.color_pattern = re.compile(r'(màu\s+)?(đỏ|xanh\s*dương|xanh\s*lá|vàng|đen|trắng|cam|tím|nâu|xám)', re.IGNORECASE)
        self.number_pattern = re.compile(r'\b(\d+)\s+(người|xe|chiếc|con)\b', re.IGNORECASE)
        self.ocr_pattern = re.compile(r'chữ\s+"([^"]+)"|biển\s*số\s+([A-Z0-9\-]+)', re.IGNORECASE)

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parses query using Regex for extreme low latency.
        """
        print(f"[Router] Fast Parsing: '{query}'")
        
        colors = [m[1].strip() for m in self.color_pattern.findall(query)]
        numbers = [f"{m[0]} {m[1]}" for m in self.number_pattern.findall(query)]
        
        ocrs = []
        for m in self.ocr_pattern.findall(query):
            ocrs.extend([g for g in m if g])
            
        entities = []
        entities.extend(colors)
        entities.extend(numbers)
        
        parsed = {
            "raw_query": query,         # Send to Vector DB (SigLIP)
            "filter_entities": entities, # Send to Milvus Payload Filter
            "filter_ocr": ocrs           # Send to ES BM25 & Milvus Payload Filter
        }
        return parsed
