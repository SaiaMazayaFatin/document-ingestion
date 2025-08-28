from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Konfigurasi project via .env (tanpa export).
    Semua nilai bisa dioverride lewat ENV saat deploy.
    """

    # =========================
    # Models
    # =========================
    # STT (Whisper)
    STT_MODEL: str = Field(default="openai/whisper-small")

    # VLM untuk Image → Text
    VLM_IMAGE_MODEL: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct")

    # Text Embedding (semua modality berujung teks)
    EMBED_MODEL: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    EMBED_DIM: int = Field(default=1024)

    # LLM untuk preprocessing (cleaning) & IE (extraction)
    CLEAN_LLM_MODEL: str = Field(default="gpt-4o-mini")
    EXTRACT_LLM_MODEL: str = Field(default="gpt-4o-mini")

    # =========================
    # Audio/Video
    # =========================
    SAMPLE_RATE: int = Field(default=16000)
    WINDOW_SECONDS: int = Field(default=30)  # Whisper window 30s
    AUDIO_OVERLAP_SECONDS: int = Field(default=2)  # overlap antar segmen untuk konteks kalimat
    STT_MAX_WORKERS: int = Field(default=2)  # paralelisme transkripsi segmen

    # =========================
    # Paths (staging data)
    # =========================
    AUDIO_FOLDER: str = Field(default="data/raw/audio")
    IMAGE_FOLDER: str = Field(default="data/raw/images")
    VIDEO_FOLDER: str = Field(default="data/raw/videos")
    DOCS_FOLDER: str = Field(default="data/raw/documents")

    # =========================
    # Vector DB (Qdrant)
    # =========================
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: str | None = Field(default=None)
    # Default collection diselaraskan dengan compose/README (rag_audio_chunks)
    QDRANT_COLLECTION: str = Field(default="rag_audio_chunks")

    # =========================
    # Graph DB (Neo4j)
    # =========================
    NEO4J_URL: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")

    # =========================
    # SQL (Postgres)
    # =========================
    PG_URL: str = Field(default="postgresql+psycopg2://postgres:postgres@localhost:5432/ragdb")

    # =========================
    # OpenAI
    # =========================
    OPENAI_API_KEY: str

    # =========================
    # Feature Flags (enable/disable subsystems quickly)
    # =========================
    ENABLE_SQL: bool = Field(default=True)
    ENABLE_QDRANT: bool = Field(default=True)
    ENABLE_NEO4J: bool = Field(default=True)
    ENABLE_EXTRACTION: bool = Field(default=True)  # extraction + graph DB

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
