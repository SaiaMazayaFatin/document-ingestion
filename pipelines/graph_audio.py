import uuid, os
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from models.schemas import PipeState, Language, SourceType
from models.metadata import build_document_meta, build_chunk_meta
from audio.preprocess import resample_to_16k_mono, split_audio_30s
from audio.stt import stt_batch_30s, merge_text
from llm.cleaning import get_clean_chain
from chunking.splitter import to_chunks
from llm.extraction import get_extract_chain
from db.sql import init_sql_engine, init_sql_schema, insert_document, insert_chunk, insert_vdb_ref, insert_triple
from db.qdrant_store import upsert_documents
from db.neo4j_store import upsert_triples
from settings import settings

def node_preprocess(state: PipeState) -> PipeState:
    out16 = f"{state['file_path']}.16k.wav"
    resample_to_16k_mono(state["file_path"], out16)
    state["file_path"] = out16
    return state

def node_stt(state: PipeState) -> PipeState:
    parts = split_audio_30s(state["file_path"])
    # simpan daftar file chunk sementara untuk dibersihkan belakangan
    state["_tmp_chunk_files"] = parts
    segs  = stt_batch_30s(parts, language=state["language"] if state["language"]!="auto" else None)
    state["transcript_raw_segments"] = segs
    state["transcript_full"] = merge_text(segs)
    return state

def node_clean(state: PipeState) -> PipeState:
    cleaned = get_clean_chain().invoke({"raw": state["transcript_full"]})
    state["transcript_clean"] = cleaned if isinstance(cleaned, str) else str(cleaned)
    return state

def node_chunk(state: PipeState) -> PipeState:
    docs = to_chunks(state["transcript_clean"], state["doc_id"], state["file_name"], state["language"])
    state["chunks"] = docs
    return state

def node_persist_vector_graph_sql(state: PipeState) -> PipeState:
    # Persist documents & chunks to SQL
    engine = init_sql_engine(); tables = init_sql_schema(engine)
    now_iso = datetime.now(timezone.utc).isoformat()

    doc_meta = build_document_meta(
        doc_id=state["doc_id"], title=state["title"],
        language=Language(state["language"]) if state["language"] in ("en","id","auto") else Language.auto,
        source=SourceType.audio_ingestion, file=state["file_name"], created_at_iso=now_iso,
        knowledge_tags=["RAG","audio","STT"], lineage={"stt_model": settings.STT_MODEL, "embed_model": settings.EMBED_MODEL}
    )
    insert_document(engine, tables["documents"], doc_meta.to_row())

    # Vector index (Qdrant)
    upsert_documents(state["chunks"])

    # SQL chunks + vdb refs
    for d in state["chunks"]:
        ch_meta = build_chunk_meta(
            chunk_id=d.metadata["chunk_id"], doc_id=d.metadata["doc_id"],
            language=Language(state["language"]) if state["language"] in ("en","id","auto") else Language.auto,
            source=SourceType.audio_ingestion, file=state["file_name"], created_at_iso=now_iso,
            segments=["auto_topic"], token_estimate=len(d.page_content.split())
        )
        insert_chunk(engine, tables["chunks"], ch_meta.to_row(text=d.page_content))
        insert_vdb_ref(engine, tables["vdb_refs"], ch_meta.chunk_id, settings.EMBED_DIM, settings.QDRANT_COLLECTION)

    # Graph extraction (per chunk) → Neo4j + SQL audit
    extractor = get_extract_chain()
    for d in state["chunks"]:
        try:
            res = extractor.invoke({"chunk": d.page_content})
        except Exception as e:  # fail-soft per chunk
            print(f"[extract][warn] skip chunk {d.metadata.get('chunk_id')} error={e}")
            continue
        try:
            upsert_triples(res.triples, doc_id=d.metadata["doc_id"], chunk_id=d.metadata["chunk_id"])
        except Exception as e:
            print(f"[neo4j][warn] skip upsert triples chunk {d.metadata.get('chunk_id')}: {e}")
        for tri in res.triples:
            if tri.confidence >= 0.8:
                try:
                    insert_triple(engine, tables["gdb_triples"], tri, d.metadata["doc_id"], d.metadata["chunk_id"])
                except Exception as e:
                    print(f"[sql][warn] skip triple audit {tri.s}-{tri.p}-{tri.o}: {e}")

    # Optional cleanup file sementara (chunk 30s + file 16k) agar storage tidak penuh
    try:
        for fp in state.get("_tmp_chunk_files", []):
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except OSError:
                pass
        # hapus file .16k.wav (bukan original)
        if state.get("file_path", "").endswith(".16k.wav") and os.path.isfile(state["file_path"]):
            os.remove(state["file_path"])
        state["_cleanup_done"] = True
    except Exception as e:
        print(f"[cleanup][warn] gagal membersihkan file sementara: {e}")
    return state

def build_graph():
    g = StateGraph(PipeState)
    g.add_node("preprocess", node_preprocess)
    g.add_node("stt", node_stt)
    g.add_node("clean", node_clean)
    g.add_node("chunk", node_chunk)
    g.add_node("persist", node_persist_vector_graph_sql)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess","stt"); g.add_edge("stt","clean"); g.add_edge("clean","chunk")
    g.add_edge("chunk","persist"); g.add_edge("persist", END)
    return g.compile()
