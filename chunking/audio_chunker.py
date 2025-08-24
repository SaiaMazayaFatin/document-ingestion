from __future__ import annotations
from typing import List
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_AUDIO_CHUNK_SIZE = 1200  # chars
DEFAULT_AUDIO_CHUNK_OVERLAP = 180


def chunk_audio(transcript: str, *, doc_id: str, file_name: str, language: str = "auto",
                chunk_size: int = DEFAULT_AUDIO_CHUNK_SIZE,
                chunk_overlap: int = DEFAULT_AUDIO_CHUNK_OVERLAP) -> List[Document]:
    """Chunk ASR transcript.

    Strategy: character recursive splitter tuned for conversational text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parts = splitter.split_text(transcript)
    now = datetime.now(timezone.utc).isoformat()
    docs: List[Document] = []
    for i, txt in enumerate(parts):
        docs.append(
            Document(
                page_content=txt,
                metadata={
                    "chunk_id": f"ch_{doc_id}_a_{i+1:03d}",
                    "doc_id": doc_id,
                    "file": file_name,
                    "source": "audio_ingestion",
                    "language": language,
                    "role_restriction": ["public_read"],
                    "created_at": now,
                    "strategy": "audio_recursive_char",
                },
            )
        )
    return docs
