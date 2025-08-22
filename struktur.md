rag-multimodal-indexing/
├─ .env
├─ requirements.txt
├─ README.md
├─ docker/
│  └─ docker-compose.yml                 # qdrant, neo4j, postgres
├─ scripts/
│  ├─ run_audio.py                       # jalankan pipeline audio saja
│  ├─ run_image.py                       # jalankan pipeline image saja
│  ├─ run_video.py                       # jalankan pipeline video saja
│  ├─ run_docs.py                        # jalankan pipeline dokumen saja
│  └─ backfill.py                        # re-index dari metadata SQL
├─ settings.py                           # Pydantic BaseSettings (semua env & model)
├─ main.py                               # dispatcher: proses semua modality
│
├─ models/
│  ├─ schemas.py                         # Pydantic: Entity, Triple, ExtractionResult, dll.
│  └─ metadata.py                        # Pydantic: DocumentMeta, ChunkMeta, validator
│
├─ common/
│  ├─ logging.py                         # logger terpusat
│  ├─ text_norm.py                       # normalisasi teks umum
│  ├─ quality.py                         # gates: dedup, PII redaction, canonicalization
│  └─ ffmpeg.py                          # helper ffmpeg (untuk video→audio)
│
├─ llm/
│  ├─ cleaning.py                        # GPT-4o-mini: cleaning transcript/teks
│  ├─ extraction.py                      # GPT-4o-mini: entity & relation extraction (structured)
│  └─ prompts/                           # template prompt
│
├─ chunking/
│  ├─ splitter.py                        # recursive + sliding window (900–1200 toks, overlap)
│  └─ splitter_semantic.py               # (opsional) semantic splitter
│
├─ embeddings/
│  └─ text_embed.py                      # Qwen/Qwen3-Embedding-0.6B wrapper (HF)
│
├─ db/
│  ├─ sql.py                             # Postgres schema: documents/chunks/vdb_refs/gdb_triples
│  ├─ qdrant_store.py                    # Qdrant upsert (text vectors only, size=1024)
│  └─ neo4j_store.py                     # Neo4j upsert triples + provenance
│
├─ loaders/
│  ├─ documents/
│  │  ├─ pdf_loader.py                   # PDF → text (LangChain PDF loader, PyMuPDF)
│  │  ├─ office_loader.py                # DOCX/PPTX → text (LangChain Office loader)
│  │  └─ html_loader.py                  # HTML → text (BeautifulSoup, LangChain)
│  ├─ images/
│  │  ├─ vlm_qwen25.py                   # Qwen2.5-VL-3B-Instruct → caption/keywords → TEXT
│  │  └─ ocr.py                          # OCR fallback → TEXT
│  └─ videos/
│     ├─ audio_extract.py                # ffmpeg: extract audio → wav 16k mono
│     └─ asr_align.py                    # (opsional) align segmen waktu
│
├─ audio/
│  ├─ preprocess.py                      # resample 16k mono, VAD, normalize, split 30s
│  ├─ stt.py                             # Whisper-small (HF) per 30 detik
│  └─ pipeline.py                        # AUDIO: preproc → STT → clean → chunk → embed(Qwen3) → Qdrant → Neo4j → SQL
│
├─ images/
│  ├─ preprocess.py                      # resize/denoise ringan (opsional)
│  ├─ parse.py                           # VLM (Qwen2.5-VL-3B) + OCR → gabung TEKS
│  └─ pipeline.py                        # IMAGE: parse→ clean → chunk → embed(Qwen3) → Qdrant → Neo4j → SQL
│
├─ videos/
│  ├─ preprocess.py                      # re-encode normalisasi (opsional), cek durasi
│  ├─ parse.py                           # extract audio → STT Whisper-small → gabung TEKS
│  └─ pipeline.py                        # VIDEO: parse→ clean → chunk → embed(Qwen3) → Qdrant → Neo4j → SQL
│
├─ documents/
│  ├─ preprocess.py                      # normalisasi teks (remove header/footer, table flatten)
│  ├─ parse.py                           # load dokumen → plain text
│  └─ pipeline.py                        # DOCS: parse→ clean → chunk → embed(Qwen3) → Qdrant → Neo4j → SQL
│
└─ pipeline/
   ├─ nodes_common.py                    # node shared: persist SQL, vector index, graph extract
   ├─ graph_audio.py                     # graph untuk audio
   ├─ graph_image.py                     # graph untuk image
   ├─ graph_video.py                     # graph untuk video
   ├─ graph_docs.py                      # graph untuk dokumen
   └─ dispatcher.py                      # memilih graph sesuai ekstensi/modality
