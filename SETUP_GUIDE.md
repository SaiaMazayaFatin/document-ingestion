# 🚀 RAG Multimodal Indexing - Complete Setup Guide

Panduan lengkap menjalankan pipeline RAG Multimodal untuk audio ingestion menggunakan Docker.

## 📋 Prerequisites

### 1. Software Requirements
- **Docker Desktop** (Windows/Mac) atau **Docker Engine** (Linux)
- **Git** untuk clone repository
- **PowerShell** (Windows) atau **Terminal** (Mac/Linux)
- Minimal **8GB RAM** dan **10GB disk space**

### 2. Verifikasi Docker Installation
```powershell
# Cek Docker tersedia
docker --version
docker compose --version

# Test Docker berjalan
docker run hello-world
```

## 🏗️ Project Structure Overview

```
rag-multimodal-indexing/
├── data/
│   └── raw/audio/          # ← Letakkan file audio (.mp3, .wav) di sini
├── docker/
│   └── docker-compose.yaml # ← Konfigurasi semua services
├── db/                     # Database connectors
├── audio/                  # Audio processing & STT
├── chunking/              # Text chunking strategies
├── pipelines/             # LangGraph workflow
└── scripts/               # Entry points
```

## 🔧 Setup Steps

### Step 1: Clone Repository
```powershell
git clone <repository-url>
cd rag-multimodal-indexing
```

### Step 2: Prepare Audio Files
```powershell
# Buat folder untuk audio input
mkdir -p data/raw/audio

# Copy file audio Anda (.mp3 atau .wav) ke folder ini
# Contoh:
# copy "C:\Downloads\podcast.mp3" data/raw/audio/
```

### Step 3: Configure Environment (Optional)
File `settings.py` sudah dikonfigurasi dengan default yang aman. Jika ingin customize:

```powershell
# Edit settings.py untuk:
# - Model STT (default: openai/whisper-small)
# - Model embedding (default: sentence-transformers/all-MiniLM-L6-v2)
# - Feature flags (ENABLE_SQL, ENABLE_QDRANT, ENABLE_NEO4J)
```

### Step 4: Start Docker Services
```powershell
cd docker

# Pull dan start semua services (database + app)
docker compose up -d

# Verifikasi semua container running
docker compose ps
```

**Expected Output:**
```
NAME            STATUS          PORTS
rag_postgres    Up              5432/tcp
rag_qdrant      Up              6333/tcp, 6334/tcp
rag_neo4j       Up              7474/tcp, 7687/tcp
```

### Step 5: Verifikasi Database Connections
```powershell
# Test semua database siap
docker compose run --rm app python test/run_tests.py --quick
```

**Expected Output:**
```
✅ PostgreSQL connection successful
✅ Qdrant connection successful  
✅ Neo4j connection successful
🎯 All systems ready!
```

## 🎵 Running Audio Ingestion

### Single Command Execution
```powershell
# Jalankan pipeline untuk semua audio di data/raw/audio/
docker compose run --rm app python scripts/run_audio.py
```

### Monitor Progress
Anda akan melihat output detail seperti:
```
[AUDIO] Processing: podcast_episode_1.mp3
[stt] loading model openai/whisper-small on device=cpu

[sql] Inserting document: doc_id=doc_abc123, title=podcast_episode_1...
[sql] ✓ Successfully inserted document: doc_abc123

[qdrant] inserted 4 docs into rag_audio_chunks

[sql] Inserting chunk: chunk_id=ch_doc_abc123_001, doc_id=doc_abc123
[sql] ✓ Successfully inserted chunk: ch_doc_abc123_001

[neo4j] Processing 5 triples for doc_id=doc_abc123, chunk_id=ch_doc_abc123_001
[neo4j] ✓ Successfully inserted 4 triples to Neo4j (filtered 1 low-confidence)

✅ Done: doc_abc123
```

## 🔍 Verifying Results

### Check Data in Databases

**PostgreSQL (Documents & Chunks):**
```powershell
docker exec -it rag_postgres psql -U postgres -d ragdb -c "
SELECT doc_id, title, created_at FROM documents ORDER BY created_at DESC LIMIT 5;
"

docker exec -it rag_postgres psql -U postgres -d ragdb -c "
SELECT chunk_id, doc_id, token_estimate FROM chunks ORDER BY created_at DESC LIMIT 5;
"
```

**Qdrant (Vector Database):**
```powershell
# Check collection status
curl -s http://localhost:6333/collections/rag_audio_chunks | jq

# Or via browser: http://localhost:6333/dashboard
```

**Neo4j (Knowledge Graph):**
```powershell
# Web interface: http://localhost:7474
# Username: neo4j, Password: neo4j123

# Query example:
# MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10;
```

### Run Test Suite
```powershell
# Comprehensive testing
docker compose run --rm app python test/run_tests.py --all

# Individual database tests
docker compose run --rm app python test/test_sql_db.py
docker compose run --rm app python test/test_vector_db.py
docker compose run --rm app python test/test_graph_db.py
```

## 🛠️ Troubleshooting

### Problem: Container Won't Start
```powershell
# Check container logs
docker logs rag_postgres
docker logs rag_qdrant  
docker logs rag_neo4j

# Restart specific service
docker compose restart postgres
```

### Problem: Out of Memory
```powershell
# Check resource usage
docker stats

# Add resource limits to docker-compose.yaml:
# deploy:
#   resources:
#     limits:
#       memory: 4G
```

### Problem: Audio Processing Fails
```powershell
# Test audio file directly
docker compose run --rm app python -c "
import librosa
try:
    y, sr = librosa.load('data/raw/audio/your_file.mp3', sr=16000)
    print(f'✅ Audio OK: duration {len(y)/sr:.1f}s')
except Exception as e:
    print(f'❌ Audio Error: {e}')
"
```

### Problem: Database Connection Fails
```powershell
# Reset all databases (WARNING: Data will be lost!)
docker compose down --volumes
docker compose up -d

# Wait for databases to initialize (30-60 seconds)
docker compose run --rm app python test/run_tests.py --quick
```

## 🧹 Maintenance Commands

### Stop All Services
```powershell
docker compose stop
```

### Restart Services
```powershell
docker compose restart
```

### View Logs
```powershell
# Real-time logs
docker compose logs -f app

# Specific service logs
docker logs rag_postgres --tail=50
```

### Clean Reset (Fresh Start)
```powershell
# ⚠️ WARNING: This deletes all data!
docker compose down --volumes
docker compose up -d
```

### Update Code Changes
```powershell
# If you modify Python code:
docker compose build app
docker compose restart app
```

## 📊 Performance Tips

### 1. GPU Acceleration (If Available)
Edit `settings.py`:
```python
# Set STT_DEVICE environment variable
STT_DEVICE = "cuda:0"  # or keep "auto" for auto-detection
```

### 2. Batch Processing
```powershell
# Process multiple files efficiently
docker compose run --rm app python scripts/run_audio.py
```

### 3. Feature Flags
Disable features you don't need in `settings.py`:
```python
ENABLE_SQL = True        # Core database
ENABLE_QDRANT = True     # Vector search
ENABLE_NEO4J = False     # Disable if not using graph
ENABLE_EXTRACTION = False # Disable if not extracting entities
```

## 🎯 Expected Results

After successful ingestion:
- **PostgreSQL**: Documents, chunks, references, triples stored
- **Qdrant**: Audio chunks embedded as vectors for semantic search
- **Neo4j**: Knowledge graph with extracted entities and relationships
- **Files**: Processed audio segments in `data/interim/`

## 📞 Support

### Common Issues
1. **Port conflicts**: Change ports in `docker-compose.yaml`
2. **Insufficient memory**: Increase Docker memory limit
3. **Model download fails**: Check internet connection
4. **Audio format unsupported**: Convert to MP3/WAV

### Debugging Commands
```powershell
# System information
docker system df
docker system info

# Container inspection
docker inspect rag_postgres
docker exec -it rag_postgres env
```

---

## 🚀 Quick Start Checklist

- [ ] Docker Desktop installed and running
- [ ] Repository cloned
- [ ] Audio files in `data/raw/audio/`
- [ ] `cd docker && docker compose up -d`
- [ ] `docker compose run --rm app python test/run_tests.py --quick`
- [ ] `docker compose run --rm app python scripts/run_audio.py`
- [ ] Verify results in databases

**Total setup time: ~10-15 minutes** ⏱️

Happy ingesting! 🎉
