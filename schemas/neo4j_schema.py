from neo4j import GraphDatabase

def create_constraints(driver):
    queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sc:Scene) REQUIRE sc.scene_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (sh:Shot) REQUIRE sh.shot_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tracklet) REQUIRE t.track_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (so:StaticObject) REQUIRE so.object_id IS UNIQUE"
    ]
    with driver.session() as session:
        for q in queries:
            session.run(q)

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
            
        # Insert Shots
        for sh in data.get("shots", []):
            session.run("""
                MERGE (sh:Shot {shot_id: $sid})
                SET sh.start_ms = $start, sh.end_ms = $end, sh.news_type = $news, sh.global_ocr = $ocr, sh.global_asr = $asr
            """, sid=sh["shot_id"], start=sh["start_ms"], end=sh["end_ms"], news=sh.get("news_type"), ocr=sh.get("global_ocr"), asr=sh.get("global_asr"))
            
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
                
        # Insert Static Objects
        for so in data.get("static_objects", []):
            session.run("""
                MERGE (so:StaticObject {object_id: $oid})
                SET so.class_label = $cls, so.ocr_text = $ocr
                WITH so
                MATCH (sh:Shot {shot_id: $shid})
                MERGE (sh)-[:CONTAINS_STATIC]->(so)
            """, oid=so["object_id"], cls=so["class_label"], ocr=so.get("ocr_text", ""), shid=so["shot_id"])
            
        # Insert Actions
        for ac in data.get("actions", []):
            session.run("""
                MERGE (a:Action {action_id: $aid})
                SET a.action_label = $label, a.confidence = $conf
                WITH a
                MATCH (t:Tracklet {track_id: $tid})
                MERGE (t)-[:PERFORMS]->(a)
            """, aid=f"action_{ac['track_id']}", label=ac["action_label"], conf=ac["confidence"], tid=ac["track_id"])
