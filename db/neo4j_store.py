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
  r.provenance = [{doc_id: $doc_id, chunk_id: $chunk_id}],
  r.confidence = $confidence,
  r.created_at = datetime()
ON MATCH SET
  r.provenance = coalesce(r.provenance, []) + {doc_id: $doc_id, chunk_id: $chunk_id},
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
    own_driver = False
    if driver is None:
        driver = init_neo4j_driver()
        own_driver = True

    try:
        with driver.session() as session:
            for t in triples:
                if getattr(t, "confidence", 0.0) < min_conf:
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
    finally:
        if own_driver:
            driver.close()

    return wrote
