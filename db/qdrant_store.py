from __future__ import annotations
from typing import Sequence, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from settings import settings


def init_qdrant() -> QdrantClient:
    """
    Ensure Qdrant collection exists (text vectors only, 1024-d, cosine).
    """
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    try:
        client.get_collection(settings.QDRANT_COLLECTION)
    except Exception:
        client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
    return client


def get_vectorstore(
    client: Optional[QdrantClient] = None,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> QdrantVectorStore:
    """
    Build a LangChain vector store bound to the configured collection.
    """
    client = client or init_qdrant()
    embeddings = embeddings or HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embeddings=embeddings,
    )


def upsert_documents(docs: Sequence[Document]) -> None:
    """
    Embed & upsert LangChain Documents into Qdrant using the configured collection.
    """
    if not docs:
        return
    vs = get_vectorstore()
    # (LangChain QdrantVectorStore akan embed internal; jika ingin validasi dimensi,
    # bisa embed manual, tapi di sini kita manfaatkan add_documents langsung.)
    try:
        vs.add_documents(list(docs))
        print(f"[qdrant] inserted {len(docs)} docs into {settings.QDRANT_COLLECTION}")
    except Exception as e:
        print(f"[qdrant][error] upsert failed: {e}")
