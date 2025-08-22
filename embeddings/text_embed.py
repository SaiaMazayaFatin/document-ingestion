from __future__ import annotations
from typing import Sequence, List
from langchain_huggingface import HuggingFaceEmbeddings
from settings import settings

_embedder: HuggingFaceEmbeddings | None = None

def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)
    return _embedder

def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    return get_embedder().embed_documents(list(texts))

def embed_query(text: str) -> List[float]:
    return get_embedder().embed_query(text)
