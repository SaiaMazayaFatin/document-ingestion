from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator

from .schemas import Language, Role, SourceType

def _parse_iso8601(ts: str) -> datetime:
    """
    Parser ISO-8601 yang toleran:
    - Menerima 'Z' → dikonversi ke +00:00
    - Memastikan hasil timezone-aware datetime
    """
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # fallback: anggap UTC kalau tidak ada tz
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

# =========================
# Document-level metadata
# =========================
class DocumentMeta(BaseModel):
    doc_id: str
    title: str
    language: Language = Language.auto
    source: SourceType
    file: str
    author: str = "unknown"
    created_at_iso: str  # ISO-8601 timestamp
    knowledge_tags: List[str] = Field(default_factory=list)
    role_restriction: List[Role] = Field(default_factory=lambda: [Role.public_read])
    lineage: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at_iso")
    @classmethod
    def _iso8601(cls, v: str) -> str:
        # Normalisasi & validasi
        dt = _parse_iso8601(v)
        return dt.isoformat()

    def to_row(self) -> Dict[str, Any]:
        created_dt = _parse_iso8601(self.created_at_iso)
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "language": self.language.value,
            "source": self.source.value,
            "file": self.file,
            "author": self.author,
            "created_at": created_dt,
            "knowledge_tags": self.knowledge_tags,
            "role_restriction": [r.value for r in self.role_restriction],
            "lineage": self.lineage,
        }

# =========================
# Chunk-level metadata
# =========================
class ChunkMeta(BaseModel):
    chunk_id: str
    doc_id: str
    language: Language
    source: SourceType
    file: str
    created_at_iso: str
    role_restriction: List[Role] = Field(default_factory=lambda: [Role.public_read])
    segments: List[str] = Field(default_factory=list)
    token_estimate: int = 0
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("token_estimate")
    @classmethod
    def _tok_nonneg(cls, v: int) -> int:
        return max(0, v)

    @field_validator("created_at_iso")
    @classmethod
    def _created_at(cls, v: str) -> str:
        dt = _parse_iso8601(v)
        return dt.isoformat()

    def to_row(self, *, text: str, segments: Optional[List[str]] = None) -> Dict[str, Any]:
        created_dt = _parse_iso8601(self.created_at_iso)
        segs = segments if segments is not None else self.segments
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "segments": segs,
            "token_estimate": self.token_estimate,
            "created_at": created_dt,
            "text": text,
        }

    def to_vdb_ref_row(self, *, vector_dim: int, collection: str) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "collection": collection,
            "vector_dim": vector_dim,
        }

# =========================
# Helper factories (opsional)
# =========================
def build_document_meta(
    *,
    doc_id: str,
    title: str,
    language: Language,
    source: SourceType,
    file: str,
    created_at_iso: str,
    author: str = "narrator",
    knowledge_tags: Optional[List[str]] = None,
    role_restriction: Optional[List[Role]] = None,
    lineage: Optional[Dict[str, Any]] = None,
) -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        title=title,
        language=language,
        source=source,
        file=file,
        author=author,
        created_at_iso=created_at_iso,
        knowledge_tags=knowledge_tags or [],
        role_restriction=role_restriction or [Role.public_read],
        lineage=lineage or {},
    )

def build_chunk_meta(
    *,
    chunk_id: str,
    doc_id: str,
    language: Language,
    source: SourceType,
    file: str,
    created_at_iso: str,
    segments: Optional[List[str]] = None,
    token_estimate: int = 0,
    role_restriction: Optional[List[Role]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ChunkMeta:
    return ChunkMeta(
        chunk_id=chunk_id,
        doc_id=doc_id,
        language=language,
        source=source,
        file=file,
        created_at_iso=created_at_iso,
        segments=segments or [],
        token_estimate=token_estimate,
        role_restriction=role_restriction or [Role.public_read],
        extra=extra or {},
    )
