# db/sql.py
from __future__ import annotations
from typing import Dict, Any, Iterable
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, Table, Column, Text, Integer, TIMESTAMP, MetaData, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import OperationalError, ProgrammingError

from settings import settings


def init_sql_engine() -> Engine:
    """
    Create SQLAlchemy engine using PG_URL from settings.
    """
    return create_engine(settings.PG_URL, future=True, pool_pre_ping=True)


def init_sql_schema(engine: Engine) -> Dict[str, Table]:
    """
    Create tables if not exist. Idempotent.
    """
    metadata = MetaData()

    documents = Table(
        "documents", metadata,
        Column("doc_id", Text, primary_key=True),
        Column("title", Text),
        Column("language", Text),
        Column("source", Text),
        Column("file", Text),
        Column("author", Text),
        Column("created_at", TIMESTAMP(timezone=True)),
        Column("knowledge_tags", ARRAY(Text)),
        Column("role_restriction", ARRAY(Text)),
        Column("lineage", JSONB),
    )

    chunks = Table(
        "chunks", metadata,
        Column("chunk_id", Text, primary_key=True),
        Column("doc_id", Text),
        Column("segments", JSONB),
        Column("token_estimate", Integer),
        Column("created_at", TIMESTAMP(timezone=True)),
        Column("text", Text),
    )

    vdb_refs = Table(
        "vdb_refs", metadata,
        Column("chunk_id", Text, primary_key=True),
        Column("collection", Text),
        Column("vector_dim", Integer),
        Column("inserted_at", TIMESTAMP(timezone=True)),
    )

    gdb_triples = Table(
        "gdb_triples", metadata,
        Column("triple_id", Text, primary_key=True),
        Column("s", Text), Column("p", Text), Column("o", Text),
        Column("doc_id", Text), Column("chunk_id", Text),
        Column("confidence", Integer),
        Column("created_at", TIMESTAMP(timezone=True)),
    )

    try:
        metadata.create_all(engine, checkfirst=True)
    except (OperationalError, ProgrammingError) as e:
        raise RuntimeError(f"Failed to create schema: {e}") from e

    return {
        "documents": documents,
        "chunks": chunks,
        "vdb_refs": vdb_refs,
        "gdb_triples": gdb_triples,
    }


@contextmanager
def tx(engine: Engine) -> Iterable[Connection]:
    """
    Simple context manager for transactional writes.
    """
    with engine.begin() as conn:
        yield conn


def insert_document(engine: Engine, documents_tbl: Table, doc: Dict[str, Any]) -> None:
    with tx(engine) as conn:
        conn.execute(documents_tbl.insert().values(**doc))


def insert_chunk(engine: Engine, chunks_tbl: Table, row: Dict[str, Any]) -> None:
    with tx(engine) as conn:
        conn.execute(chunks_tbl.insert().values(**row))


def insert_vdb_ref(
    engine: Engine,
    vdb_tbl: Table,
    chunk_id: str,
    dim: int,
    collection: str,
) -> None:
    with tx(engine) as conn:
        conn.execute(
            vdb_tbl.insert().values(
                chunk_id=chunk_id,
                collection=collection,
                vector_dim=dim,
                inserted_at=datetime.now(timezone.utc),
            )
        )


def insert_triple(
    engine: Engine,
    gdb_tbl: Table,
    tri: Any,  # expects object with .s, .p, .o, .confidence
    doc_id: str,
    chunk_id: str,
) -> None:
    from uuid import uuid4
    with tx(engine) as conn:
        conn.execute(
            gdb_tbl.insert().values(
                triple_id=str(uuid4()),
                s=tri.s,
                p=tri.p,
                o=tri.o,
                doc_id=doc_id,
                chunk_id=chunk_id,
                confidence=int(tri.confidence * 100),
                created_at=datetime.now(timezone.utc),
            )
        )
