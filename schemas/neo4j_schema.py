from neo4j import GraphDatabase
import hashlib

def create_constraints(driver):
    queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sc:Scene) REQUIRE sc.scene_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sh:Shot) REQUIRE sh.shot_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tracklet) REQUIRE t.track_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (so:StaticObject) REQUIRE so.object_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (l:LexicalNode) REQUIRE l.text_id IS UNIQUE"
    ]
    with driver.session() as session:
        for q in queries:
            session.run(q)

def _hash_text(text: str, prefix: str) -> str:
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"

def ingest_graph(driver, data):
    """
    data should contain: video_metadata, scenes, shots, tracklets, static_objects, actions, shot_to_tracklets
    """
    with driver.session() as session:
        # Insert Video
        v = data.get("video_metadata", {})
        session.run("MERGE (v:Video {video_id: $vid}) SET v.duration_ms = $dur, v.fps = $fps", 
                    vid=v.get("video_id"), dur=v.get("duration_ms"), fps=v.get("fps"))
                    
        # Insert Scenes
        for sc in data.get("scenes", []):
            session.run("""
                MERGE (sc:Scene {scene_id: $sid})
                SET sc.start_ms = $start, sc.end_ms = $end, sc.news_type = $news
                WITH sc
                MATCH (v:Video {video_id: $vid})
                MERGE (v)-[:HAS_SCENE]->(sc)
            """, sid=sc["scene_id"], start=sc["start_ms"], end=sc["end_ms"], news=sc.get("news_type"), vid=v.get("video_id"))
            
        # Insert Shots & LexicalNodes (Global)
        for sh in data.get("shots", []):
            session.run("""
                MERGE (sh:Shot {shot_id: $sid})
                SET sh.start_ms = $start, sh.end_ms = $end, sh.news_type = $news, sh.keyframe_path = $kp
            """, sid=sh["shot_id"], start=sh["start_ms"], end=sh["end_ms"], news=sh.get("news_type"), kp=sh.get("image_vector_path", ""))
            
            # Extract OCR text node
            if sh.get("global_ocr"):
                ocr_txt = sh["global_ocr"]
                text_id = _hash_text(ocr_txt, f"ocr_{sh['shot_id']}")
                session.run("""
                    MERGE (l:LexicalNode {text_id: $tid})
                    SET l.text = $txt, l.source = 'global_ocr'
                    WITH l
                    MATCH (sh:Shot {shot_id: $shid})
                    MERGE (sh)-[:HAS_TEXT]->(l)
                """, tid=text_id, txt=ocr_txt, shid=sh["shot_id"])
                
            # Extract ASR text node
            if sh.get("global_asr"):
                asr_txt = sh["global_asr"]
                text_id = _hash_text(asr_txt, f"asr_{sh['shot_id']}")
                session.run("""
                    MERGE (l:LexicalNode {text_id: $tid})
                    SET l.text = $txt, l.source = 'global_asr'
                    WITH l
                    MATCH (sh:Shot {shot_id: $shid})
                    MERGE (sh)-[:HAS_TEXT]->(l)
                """, tid=text_id, txt=asr_txt, shid=sh["shot_id"])
            
        # Link Scenes to Shots
        for sc in data.get("scenes", []):
            for sh_id in sc.get("shot_ids", []):
                session.run("""
                    MATCH (sc:Scene {scene_id: $scid}), (sh:Shot {shot_id: $shid})
                    MERGE (sc)-[:HAS_SHOT]->(sh)
                """, scid=sc["scene_id"], shid=sh_id)

        # Insert Tracklets
        for tr in data.get("tracklets", []):
            session.run("""
                MERGE (t:Tracklet {track_id: $tid})
                SET t.class_label = $cls, t.start_ms = $start, t.end_ms = $end
            """, tid=tr["track_id"], cls=tr["class_label"], start=tr["start_ms"], end=tr["end_ms"])
            
        # Link Shots to Tracklets
        for sh_id, track_ids in data.get("shot_to_tracklets", {}).items():
            for tid in track_ids:
                session.run("""
                    MATCH (sh:Shot {shot_id: $shid}), (t:Tracklet {track_id: $tid})
                    MERGE (sh)-[:CONTAINS_TRACKLET]->(t)
                """, shid=sh_id, tid=tid)
                
        # Insert Static Objects & LexicalNodes (Local)
        for so in data.get("static_objects", []):
            session.run("""
                MERGE (so:StaticObject {object_id: $oid})
                SET so.class_label = $cls
                WITH so
                MATCH (sh:Shot {shot_id: $shid})
                MERGE (sh)-[:CONTAINS_STATIC]->(so)
            """, oid=so["object_id"], cls=so["class_label"], shid=so["shot_id"])
            
            if so.get("ocr_text"):
                ocr_txt = so["ocr_text"]
                text_id = _hash_text(ocr_txt, f"loc_{so['object_id']}")
                session.run("""
                    MERGE (l:LexicalNode {text_id: $tid})
                    SET l.text = $txt, l.source = 'local_ocr'
                    WITH l
                    MATCH (so:StaticObject {object_id: $soid})
                    MERGE (so)-[:HAS_TEXT]->(l)
                """, tid=text_id, txt=ocr_txt, soid=so["object_id"])
            
        # Insert Events (formerly Actions)
        for ac in data.get("actions", []):
            session.run("""
                MERGE (e:Event {event_id: $eid})
                SET e.action_label = $label, e.confidence = $conf
                WITH e
                MATCH (t:Tracklet {track_id: $tid})
                MERGE (t)-[:PERFORMS]->(e)
            """, eid=ac.get("event_id", f"event_{ac['track_id']}"), label=ac["action_label"], conf=ac["confidence"], tid=ac["track_id"])
