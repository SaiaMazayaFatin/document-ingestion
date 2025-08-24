import uuid, os
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from models.schemas import PipeState, Language, SourceType
from models.metadata import build_document_meta, build_chunk_meta
from audio.preprocess import resample_to_16k_mono, split_audio_with_overlap, transcribe_segments_parallel, merge_overlap_text
from audio.stt import _get_asr
from llm.cleaning import get_clean_chain
from chunking.dispatcher import dispatch_chunk
from llm.extraction import get_extract_chain
from db.sql import init_sql_engine, init_sql_schema, insert_document, insert_chunk, insert_vdb_ref, insert_triple
from db.qdrant_store import upsert_documents
from db.neo4j_store import upsert_triples
from settings import settings

def node_preprocess(state: PipeState) -> PipeState:
    # Simpan file resample ke folder interim agar raw tetap bersih
    base_dir = os.path.dirname(state["file_path"])
    interim_dir = "data/interim/audio"
    os.makedirs(interim_dir, exist_ok=True)
    
    # Clean filename for intermediate files
    original_name = os.path.splitext(os.path.basename(state["file_path"]))[0]
    out16 = os.path.join(interim_dir, f"{original_name}_16k.wav")
    
    resample_to_16k_mono(state["file_path"], out16)
    state["file_path"] = out16
    return state

def node_stt(state: PipeState) -> PipeState:
    parts = split_audio_with_overlap(state["file_path"])
    state["_tmp_chunk_files"] = parts
    asr = _get_asr()
    def _one(pth, lang):
        if lang:
            out = asr(pth, generate_kwargs={"language": lang})
        else:
            out = asr(pth)
        return {"file": pth, "text": out.get("text", ""), "chunks": out.get("chunks")}
    language = state["language"] if state["language"] != "auto" else None
    segs = transcribe_segments_parallel(parts, _one, language)
    state["transcript_raw_segments"] = segs
    state["transcript_full"] = merge_overlap_text(segs)
    return state

def node_clean(state: PipeState) -> PipeState:
    cleaned = get_clean_chain().invoke({"raw": state["transcript_full"]})
    state["transcript_clean"] = cleaned if isinstance(cleaned, str) else str(cleaned)
    return state

def node_chunk(state: PipeState) -> PipeState:
    # Use modality-specific dispatcher (audio graph => modality fixed to 'audio')
    docs = dispatch_chunk(modality="audio", raw_text=state["transcript_clean"], doc_id=state["doc_id"], file_name=state["file_name"], language=state["language"])
    state["chunks"] = docs
    return state

def node_persist_vector_graph_sql(state: PipeState) -> PipeState:
    now_iso = datetime.now(timezone.utc).isoformat()
    engine = None; tables = None
    if settings.ENABLE_SQL:
        try:
            engine = init_sql_engine(); tables = init_sql_schema(engine)
        except Exception as e:
            print(f"[sql][error] init failed -> disable SQL: {e}"); engine = None

    # Document row
    if settings.ENABLE_SQL and engine and tables:
        try:
            doc_meta = build_document_meta(
                doc_id=state["doc_id"], title=state["title"],
                language=Language(state["language"]) if state["language"] in ("en","id","auto") else Language.auto,
                source=SourceType.audio_ingestion, file=state["file_name"], created_at_iso=now_iso,
                knowledge_tags=["RAG","audio","STT"], lineage={"stt_model": settings.STT_MODEL, "embed_model": settings.EMBED_MODEL}
            )
            insert_document(engine, tables["documents"], doc_meta.to_row())
        except Exception as e:
            print(f"[sql][warn] skip document insert: {e}")

    # Vector DB upsert
    if settings.ENABLE_QDRANT:
        try:
            upsert_documents(state["chunks"])
        except Exception as e:
            print(f"[qdrant][error] upsert failed: {e}")

    # Chunks + refs
    if settings.ENABLE_SQL and engine and tables:
        for d in state["chunks"]:
            try:
                ch_meta = build_chunk_meta(
                    chunk_id=d.metadata["chunk_id"], doc_id=d.metadata["doc_id"],
                    language=Language(state["language"]) if state["language"] in ("en","id","auto") else Language.auto,
                    source=SourceType.audio_ingestion, file=state["file_name"], created_at_iso=now_iso,
                    segments=["auto_topic"], token_estimate=len(d.page_content.split())
                )
                insert_chunk(engine, tables["chunks"], ch_meta.to_row(text=d.page_content))
                insert_vdb_ref(engine, tables["vdb_refs"], ch_meta.chunk_id, settings.EMBED_DIM, settings.QDRANT_COLLECTION)
            except Exception as e:
                print(f"[sql][warn] skip chunk {d.metadata.get('chunk_id')}: {e}")

    # Extraction + Neo4j
    if settings.ENABLE_EXTRACTION:
        try:
            extractor = get_extract_chain()
        except Exception as e:
            print(f"[extract][error] chain init failed -> skip extraction: {e}"); extractor = None
        if extractor:
            for d in state["chunks"]:
                try:
                    res = extractor.invoke({"chunk": d.page_content})
                except Exception as e:
                    print(f"[extract][warn] skip chunk {d.metadata.get('chunk_id')} error={e}")
                    continue
                if settings.ENABLE_NEO4J:
                    try:
                        upsert_triples(res.triples, doc_id=d.metadata["doc_id"], chunk_id=d.metadata["chunk_id"])
                    except Exception as e:
                        print(f"[neo4j][warn] skip triples {d.metadata.get('chunk_id')}: {e}")
                if settings.ENABLE_SQL and engine and tables:
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
        # hapus file _16k.wav (bukan original)
        if state.get("file_path", "").endswith("_16k.wav") and os.path.isfile(state["file_path"]):
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
