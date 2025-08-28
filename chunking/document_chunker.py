from __future__ import annotations
from typing import List
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DOC_CHUNK_SIZE = 1600
DEFAULT_DOC_CHUNK_OVERLAP = 200

# For documents we often want to bias toward paragraph & sentence boundaries first.


def chunk_document(text: str, *, doc_id: str, file_name: str, language: str = "auto",
                   chunk_size: int = DEFAULT_DOC_CHUNK_SIZE,
                   chunk_overlap: int = DEFAULT_DOC_CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    parts = splitter.split_text(text)
    now = datetime.now(timezone.utc).isoformat()
    docs: List[Document] = []
    for i, p in enumerate(parts):
        docs.append(
            Document(
                page_content=p,
                metadata={
                    "chunk_id": f"ch_{doc_id}_d_{i+1:03d}",
                    "doc_id": doc_id,
                    "file": file_name,
                    "source": "document_ingestion",
                    "language": language,
                    "role_restriction": ["public_read"],
                    "created_at": now,
                    "strategy": "document_recursive_char",
                },
            )
        )
    return docs
