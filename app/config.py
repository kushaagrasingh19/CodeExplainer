import os
from typing import Optional


class Settings:
    def __init__(self) -> None:
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_embedding_model: str = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
        )
        self.chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./storage/chroma")
        self.max_context_docs: int = int(os.getenv("MAX_CONTEXT_DOCS", "4"))
        self.collection_name: str = os.getenv("CHROMA_COLLECTION", "code-explainer")
        self.temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))


def get_settings() -> Settings:
    return Settings()