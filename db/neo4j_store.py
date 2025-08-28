from __future__ import annotations
from typing import Sequence

from neo4j import GraphDatabase, Driver
from settings import settings

# Simple, generic triple upsert with provenance and confidence
NEO4J_MERGE = """
MERGE (s:Entity {name: $s})
MERGE (o:Entity {name: $o})
MERGE (s)-[r:REL {predicate: $p}]->(o)
ON CREATE SET
  r.doc_ids = [$doc_id],
  r.chunk_ids = [$chunk_id],
  r.confidence = $confidence,
  r.created_at = datetime()
ON MATCH SET
  r.doc_ids = CASE WHEN NOT $doc_id IN r.doc_ids THEN r.doc_ids + $doc_id ELSE r.doc_ids END,
  r.chunk_ids = CASE WHEN NOT $chunk_id IN r.chunk_ids THEN r.chunk_ids + $chunk_id ELSE r.chunk_ids END,
  r.confidence = CASE WHEN r.confidence < $confidence THEN $confidence ELSE r.confidence END
"""


def init_neo4j_driver() -> Driver:
    """
    Create a Neo4j driver from settings.
    """
    return GraphDatabase.driver(
        settings.NEO4J_URL,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def upsert_triples(
    triples: Sequence,  # expects objects with .s, .p, .o, .confidence
    doc_id: str,
    chunk_id: str,
    min_conf: float = 0.8,
    driver: Driver | None = None,
) -> int:
    """
    Upsert triples with a minimum confidence threshold.
    Returns number of triples written.
    """
    wrote = 0
    filtered = 0
    own_driver = False
    if driver is None:
        driver = init_neo4j_driver()
        own_driver = True

    print(f"[neo4j] Processing {len(triples)} triples for doc_id={doc_id}, chunk_id={chunk_id}")
    
    try:
        with driver.session() as session:
            for t in triples:
                if getattr(t, "confidence", 0.0) < min_conf:
                    filtered += 1
                    continue
                session.run(
                    NEO4J_MERGE,
                    s=t.s,
                    p=t.p,
                    o=t.o,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    confidence=float(t.confidence),
                )
                wrote += 1
        
        print(f"[neo4j] ✓ Successfully inserted {wrote} triples to Neo4j (filtered {filtered} low-confidence)")
        if wrote > 0:
            print(f"[neo4j] Sample triple: {triples[0].s} --[{triples[0].p}]--> {triples[0].o}")
            
    except Exception as e:
        print(f"[neo4j] ✗ Error inserting triples: {e}")
        raise
    finally:
        if own_driver:
            driver.close()

    return wrote
