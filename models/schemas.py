from __future__ import annotations
from typing import List, Dict, Any, TypedDict
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class Modality(str, Enum):
    audio = "audio"
    image = "image"
    video = "video"
    document = "document"

class Language(str, Enum):
    auto = "auto"
    id = "id"
    en = "en"

class Role(str, Enum):
    public_read = "public_read"
    internal = "internal"
    restricted = "restricted"

class SourceType(str, Enum):
    audio_ingestion = "audio_ingestion"
    image_ingestion = "image_ingestion"
    video_ingestion = "video_ingestion"
    document_ingestion = "document_ingestion"

class Entity(BaseModel):
    name: str
    aliases: List[str] = Field(default_factory=list)

class Triple(BaseModel):
    s: str
    p: str
    o: str
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    triples: List[Triple] = Field(default_factory=list)

class PipeState(TypedDict, total=False):
    modality: str
    doc_id: str
    title: str
    language: str
    file_path: str
    file_name: str

    # audio/video
    transcript_raw_segments: List[Dict[str, Any]]
    transcript_full: str
    transcript_clean: str

    # image/document
    raw_text: str
    cleaned_text: str

    # hasil umum
    chunks: List[Document]
    extraction: Dict[str, Any]
