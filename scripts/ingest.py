import argparse
from typing import List, Optional

from app.rag.indexer import ingest_path


def parse_extensions(ext_csv: Optional[str]) -> Optional[List[str]]:
    if not ext_csv:
        return None
    parts = [p.strip().lower() for p in ext_csv.split(",") if p.strip()]
    return parts or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a directory of code into ChromaDB.")
    parser.add_argument("--path", required=True, help="Directory path to index")
    parser.add_argument("--extensions", default=None, help="Comma-separated extensions, e.g., py,js,ts")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for vectorstore add")
    args = parser.parse_args()

    exts = parse_extensions(args.extensions)
    count = ingest_path(args.path, extensions=exts, batch_size=args.batch_size)
    print(f"Indexed {count} chunks into ChromaDB.")


if __name__ == "__main__":
    main()