from __future__ import annotations
from typing import List
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_VIDEO_CHUNK_SIZE = 1400
DEFAULT_VIDEO_CHUNK_OVERLAP = 180

# Video transcripts may have scene transitions; future: inject scene boundaries.

def chunk_video(transcript: str, *, doc_id: str, file_name: str, language: str = "auto",
                chunk_size: int = DEFAULT_VIDEO_CHUNK_SIZE,
                chunk_overlap: int = DEFAULT_VIDEO_CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parts = splitter.split_text(transcript)
    now = datetime.now(timezone.utc).isoformat()
    docs: List[Document] = []
    for i, t in enumerate(parts):
        docs.append(
            Document(
                page_content=t,
                metadata={
                    "chunk_id": f"ch_{doc_id}_v_{i+1:03d}",
                    "doc_id": doc_id,
                    "file": file_name,
                    "source": "video_ingestion",
                    "language": language,
                    "role_restriction": ["public_read"],
                    "created_at": now,
                    "strategy": "video_recursive_char",
                },
            )
        )
    return docs
