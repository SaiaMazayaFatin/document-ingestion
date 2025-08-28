from __future__ import annotations
"""Chunking strategy dispatcher for different modalities.

Provides a single function `dispatch_chunk` that selects the correct
chunking implementation based on modality.

This decouples graph node logic from concrete strategy modules.
"""
from typing import List, Literal
from langchain_core.documents import Document

from .audio_chunker import chunk_audio
from .document_chunker import chunk_document
from .image_chunker import chunk_image
from .video_chunker import chunk_video

Modality = Literal["audio", "document", "image", "video"]


def dispatch_chunk(*, modality: Modality, raw_text: str, doc_id: str, file_name: str, language: str = "auto") -> List[Document]:
    if modality == "audio":
        return chunk_audio(raw_text, doc_id=doc_id, file_name=file_name, language=language)
    if modality == "document":
        return chunk_document(raw_text, doc_id=doc_id, file_name=file_name, language=language)
    if modality == "image":
        return chunk_image(raw_text, doc_id=doc_id, file_name=file_name, language=language)
    if modality == "video":
        return chunk_video(raw_text, doc_id=doc_id, file_name=file_name, language=language)
    raise ValueError(f"Unsupported modality: {modality}")
