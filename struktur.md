rag-multimodal-indexing/
├─ .env
├─ requirements.txt
├─ README.md
├─ struktur.md
├─ docker/
│  └─ docker-compose.yml                # qdrant, neo4j, postgres
├─ scripts/
│  ├─ run_audio.py                      # jalankan graph audio
│  ├─ run_image.py
│  ├─ run_video.py
│  ├─ run_docs.py
│  └─ backfill.py                       # re-index dari metadata SQL
├─ settings.py                          # Pydantic BaseSettings
├─ main.py                              # dispatcher multi-modality (invoke graph)
│
├─ models/
│  ├─ schemas.py                        # Enums, PipeState, IE schemas
│  └─ metadata.py                       # DocumentMeta, ChunkMeta
│
├─ common/
│  ├─ logging.py
│  ├─ text_norm.py
│  ├─ quality.py
│  └─ ffmpeg.py
│
├─ llm/
│  ├─ cleaning.py
│  ├─ extraction.py
│  └─ prompts/
│
├─ chunking/
│  ├─ audio_chunker.py
│  ├─ document_chunker.py
│  ├─ image_chunker.py
│  ├─ video_chunker.py
│  └─ dispatcher.py                     # pilih strategi per modality
│
├─ embeddings/                          # (dihapus - integrasi langsung di Qdrant)
│  └─ (deprecated)
│
├─ db/
│  ├─ sql.py                            # Postgres (documents, chunks, vdb_refs, gdb_triples)
│  ├─ qdrant_store.py                   # Qdrant upsert
│  └─ neo4j_store.py                    # Neo4j triples
│
├─ audio/
│  ├─ preprocess.py                     # resample 16k mono, split 30s
│  ├─ stt.py                            # ASR (cached pipeline)
│  └─ parse.py                          # (wrapper) resample + split + STT + merge
│
├─ images/
│  ├─ preprocess.py                     # resize/denoise (opsional)
│  └─ parse.py                          # VLM + OCR → text
│
├─ videos/
│  ├─ preprocess.py                     # re-encode / metadata (opsional)
│  └─ parse.py                          # extract audio → STT → text
│
├─ documents/
│  ├─ preprocess.py                     # normalisasi (header/footer, tables)
│  └─ parse.py                          # pdf/docx/html → text
│
├─ pipelines/
│  ├─ nodes_common.py                   # node persist (SQL, Qdrant, Neo4j)
│  ├─ graph_audio.py
│  ├─ graph_image.py
│  ├─ graph_video.py
│  ├─ graph_docs.py
│  └─ dispatcher.py                     # pilih graph sesuai ekstensi
│
├─ data/                                # contoh data; struktur real-time
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
│
└─ .gitignore
