# document-ingestion

Pipeline ingestion multimodal (fokus audio saat ini) dengan LangGraph + persistence ke Postgres, Qdrant, dan Neo4j (opsional lewat feature flags).

## 1. Arsitektur Audio (Graph)
Graph utama: `pipelines/graph_audio.py`

Tahapan node (ringkas):
1. Preprocess (`audio/preprocess.py`)
	- Resample → 16k mono.
	- Deteksi / fallback ffmpeg (jika tidak ada, pakai slicing librosa).
2. Segment + Overlap
	- Pecah audio jadi window 30s (default `WINDOW_SECONDS`) dengan overlap 2s (`AUDIO_OVERLAP_SECONDS`).
3. STT Parallel (`audio/stt.py` + helper di preprocess)
	- Transkripsi tiap segmen (pipeline Whisper / model HF) → parallel workers (`STT_MAX_WORKERS`).
	- Auto pilih device (CUDA kalau tersedia).
4. Cleaning (`llm/cleaning.py`)
	- Normalisasi & perapian teks.
5. Chunking (Modality dispatcher)
	- Audio → `chunking/audio_chunker.py` melalui dispatcher (`chunking/dispatcher.py`).
6. Extraction (opsional) (`llm/extraction.py`)
	- Triple / struktur untuk graph (dipakai jika `ENABLE_EXTRACTION=true`).
7. Persist (conditional via flags)
	- SQL: `db/sql.py`
	- Qdrant: `db/qdrant_store.py`
	- Neo4j: `db/neo4j_store.py`
8. Cleanup
	- Segment file sementara dihapus.

Legacy `audio/pipeline.py` & `chunking/splitter.py` telah dihapus.

## 2. Feature Flags
Atur cepat subsistem di `.env`:
- `ENABLE_SQL=true|false`
- `ENABLE_QDRANT=true|false`
- `ENABLE_NEO4J=true|false`
- `ENABLE_EXTRACTION=true|false` (bergantung graph + Neo4j)

## 3. .env Contoh (Root Proyek)
```
OPENAI_API_KEY=sk-REPLACE
PG_URL=postgresql+psycopg2://postgres:postgres@postgres:5432/ragdb
QDRANT_URL=http://qdrant:6333
NEO4J_URL=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
QDRANT_COLLECTION=rag_audio_chunks
STT_MODEL=openai/whisper-small
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBED_DIM=1024
ENABLE_SQL=true
ENABLE_QDRANT=true
ENABLE_NEO4J=true
ENABLE_EXTRACTION=true
```

Untuk mode lokal (tanpa Docker) ubah host ke `localhost` (PG_URL, QDRANT_URL, NEO4J_URL).

## 4. Menjalankan (Mode Docker)
Prereq: Docker Desktop aktif.

Letakkan file audio (.mp3/.wav) di `data/raw/audio/`.

Quick start:
```powershell
cd docker
docker compose up -d   # jalankan semua service
docker compose run --rm app python scripts/run_audio.py  # ingestion manual
```

Jalankan sekali (biarkan app container eksekusi default command):
```powershell
docker logs -f rag_audio_app
```

Re-run setelah tambah audio baru:
```powershell
docker compose run --rm app python scripts/run_audio.py
```

Force rebuild (jika ganti Dockerfile / requirements):
```powershell
cd docker
docker compose build app
docker compose up -d app
```

## 5. Menjalankan (Mode Lokal Tanpa Docker)
```powershell
# 1. Virtual env
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. (Opsional) install ffmpeg
winget install --id=Gyan.FFmpeg

# 4. Pastikan Postgres, Qdrant, Neo4j hidup (manual / container terpisah)

# 5. Jalankan pipeline
python scripts/run_audio.py
```

## 6. Verifikasi Hasil
Postgres (tabel terbentuk):
```powershell
docker exec -it rag_postgres psql -U postgres -d ragdb -c "\dt"
```
Qdrant collections:
```powershell
curl http://localhost:6333/collections
```
Neo4j (browser http://localhost:7474):
```cypher
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 10;
```

## 7. Struktur Direktori Relevan
- `audio/` preprocessing & STT helpers
- `chunking/` chunker per modality + dispatcher
- `llm/` cleaning & extraction prompt chains
- `db/` penyimpanan (SQL / Qdrant / Neo4j)
- `pipelines/graph_audio.py` definisi LangGraph
- `scripts/run_audio.py` entry ingestion batch
- `settings.py` konfigurasi + feature flags

## 8. Troubleshooting Cepat
| Gejala | Penyebab | Solusi |
|--------|----------|--------|
| ModuleNotFoundError (pydantic) | Command default di-override sebelum pip install | Jalankan `docker compose run --rm app` atau pakai Dockerfile build |
| Connection refused Postgres/Qdrant/Neo4j | Service belum siap | Cek `docker ps`, jalankan ulang atau gunakan start bertahap |
| OPENAI_API_KEY kosong di container | `.env` tidak terbaca path | Pastikan `.env` di root dan compose `env_file: ../.env` |
| Dimensi embedding mismatch | EMBED_DIM tidak cocok model | Set `EMBED_DIM` sesuai model (cth 1024) |
| Lambat STT audio panjang | Worker sedikit | Naikkan `STT_MAX_WORKERS` (hati‑hati CPU) |
| ffmpeg missing warning | ffmpeg tidak terinstall | Install ffmpeg agar segmentasi lebih akurat |

## 9. Pengembangan Lanjut (Ide)
- Healthcheck untuk Postgres/Qdrant/Neo4j + depends_on:condition.
- Deduplicate teks overlap (merge boundary heuristics).
- Graph ingestion untuk image/video/doc (menggunakan dispatcher sudah siap — tinggal graph terpisah).
- Caching hasil extraction untuk chunk yang sama (idempotensi).

## 10. Catatan Tambahan
- Confidence triple disimpan 0–100 di SQL; internal 0–1.
- Pipeline fail‑soft: error extraction/persist tidak hentikan seluruh proses (dicatat di log).

---
Untuk pertanyaan atau perlu mode retrieval/query example, tambahkan issue atau minta contoh lanjutan.