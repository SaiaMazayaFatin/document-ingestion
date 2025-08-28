from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime, timezone
from typing import List

def to_chunks(text: str, doc_id: str, file_name: str, language: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180, separators=["\n\n","\n",". "," ",""])
    pieces = splitter.split_text(text)
    now = datetime.now(timezone.utc).isoformat()
    return [Document(page_content=t, metadata={
        "chunk_id": f"ch_{doc_id}_{i+1:02d}", "doc_id": doc_id,
        "file": file_name, "source": "audio_ingestion",
        "language": language, "role_restriction": ["public_read"],
        "created_at": now
    }) for i, t in enumerate(pieces)]
