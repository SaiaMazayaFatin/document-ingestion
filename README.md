# document-ingestion

## Audio Ingestion Pipeline

Alur utama audio menggunakan graph: `pipelines/graph_audio.py`

Tahapan:
1. Preprocess: resample → 16k mono (`audio/preprocess.py`).
2. Split: pecah audio menjadi potongan 30s.
3. STT: transkripsi tiap potongan (HF ASR, model: `settings.STT_MODEL`) dengan cache (`audio/stt.py`).
4. Cleaning: normalisasi teks (LLM chain `llm/cleaning.py`).
5. Chunking: pemotongan teks jadi dokumen kecil (`chunking/splitter.py`).
6. Persist:
	- SQL (dokumen, chunks, referensi vektor, triples audit) via `db/sql.py`.
	- Vector store (Qdrant) via `db/qdrant_store.py`.
	- Knowledge graph (Neo4j) via `db/neo4j_store.py` + extraction LLM (`llm/extraction.py`).
7. Cleanup: file sementara (.16k.wav + chunk 30s) dihapus otomatis.

File legacy `audio/pipeline.py` telah dihapus untuk mengurangi duplikasi & kebingungan. Gunakan graph sebagai satu-satunya jalur eksekusi.

## Menjalankan (contoh)
Script entry (audio): `scripts/run_audio.py` membangun graph dan memproses file di `data/raw/audio/`.

## Konfigurasi
Semua variabel (model names, DB, Qdrant, Neo4j) diatur lewat `.env` yang dibaca oleh `settings.py` (Pydantic BaseSettings).

## Komponen Lain
- `models/metadata.py`: struktur DocumentMeta & ChunkMeta.
- `models/schemas.py`: enumerasi & `PipeState`.
- `pipelines/graph_audio.py`: definisi node & edges LangGraph.

## Catatan
- Extraction triple fail-soft per chunk (tidak hentikan pipeline bila 1 chunk error).
- Confidence triple tersimpan 0–100 di SQL, tapi 0–1 di memori.
- Tambahkan test sesuai kebutuhan untuk tiap node agar regresi mudah terdeteksi.