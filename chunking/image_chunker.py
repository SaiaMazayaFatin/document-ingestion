from __future__ import annotations
from typing import List
from datetime import datetime, timezone
from langchain_core.documents import Document

# Images generally yield shorter caption + OCR text; we can group them as one or few chunks.

def chunk_image(caption_text: str, *, doc_id: str, file_name: str, language: str = "auto") -> List[Document]:
    text = caption_text.strip()
    now = datetime.now(timezone.utc).isoformat()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "chunk_id": f"ch_{doc_id}_i_001",
                "doc_id": doc_id,
                "file": file_name,
                "source": "image_ingestion",
                "language": language,
                "role_restriction": ["public_read"],
                "created_at": now,
                "strategy": "image_single",
            },
        )
    ]
