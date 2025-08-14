import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import get_settings


_embeddings_cache: Optional[OpenAIEmbeddings] = None
_vectorstore_cache: Optional[Chroma] = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        settings = get_settings()
        _embeddings_cache = OpenAIEmbeddings(model=settings.openai_embedding_model)
    return _embeddings_cache


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_vectorstore() -> Chroma:
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    settings = get_settings()
    ensure_dir(settings.chroma_db_dir)

    embeddings = get_embeddings()

    _vectorstore_cache = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_dir,
    )
    return _vectorstore_cache