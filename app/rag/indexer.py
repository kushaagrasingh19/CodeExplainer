import os
from typing import Iterable, List, Optional, Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.rag.vectorstore import get_vectorstore


DEFAULT_EXTENSIONS = {
    "py", "js", "ts", "tsx", "jsx", "java", "kt", "go", "rs", "rb", "php",
    "c", "h", "cpp", "hpp", "cs", "scala", "swift", "m", "mm",
    "sql", "md", "yml", "yaml", "toml", "ini", "json"
}


def iter_files(root_path: str, extensions: Optional[Sequence[str]] = None) -> Iterable[str]:
    allowed = set((extensions or DEFAULT_EXTENSIONS))
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip common vendor or build dirs
        if any(skip in dirpath.lower() for skip in [".git", "node_modules", "dist", "build", "venv", ".venv", "__pycache__", ".cache"]):
            continue
        for filename in filenames:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext in allowed:
                yield os.path.join(dirpath, filename)


def load_documents(file_paths: Sequence[str]) -> List[Document]:
    documents: List[Document] = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        metadata = {"source": path}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def chunk_documents(documents: Sequence[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", "",],
    )
    return splitter.split_documents(list(documents))


def ingest_path(path: str, extensions: Optional[Sequence[str]] = None, batch_size: int = 128) -> int:
    file_paths = list(iter_files(path, extensions))
    if not file_paths:
        return 0

    docs = load_documents(file_paths)
    chunks = chunk_documents(docs)

    vs = get_vectorstore()

    # Add in batches to reduce memory overhead
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vs.add_documents(batch)
        total += len(batch)
    vs.persist()
    return total