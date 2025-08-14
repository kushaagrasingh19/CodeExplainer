# AI Code Explainer (LangChain + FastAPI + ChromaDB + OpenAI)

An agentic code explainer that understands any code block and explains it instantly using LangChain, FastAPI, and a RAG pipeline backed by ChromaDB. Uses your OpenAI API key for LLM and embeddings.

## Features
- Explain arbitrary code blocks with structured, actionable insights
- Retrieval-Augmented Generation (RAG) over your codebase using ChromaDB
- Lightweight FastAPI service with a single `/explain` endpoint
- Optional ingestion script to index any local repo or folder

## Quickstart

### 1) Prereqs
- Python 3.10+
- An OpenAI API key

### 2) Setup
```bash
cd /workspace
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to include your key
```

### 3) Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

Open the interactive docs at: `http://127.0.0.1:8000/docs`

### 4) Ingest your codebase into Chroma (optional but recommended)
```bash
# Index a directory into the vector store
python scripts/ingest.py --path /path/to/your/repo --extensions py,js,ts,tsx,java,go,rs,cpp,c,h,md
```

The vector store will be persisted under `./storage/chroma` by default.

### 5) Example request
```bash
curl -s -X POST http://127.0.0.1:8000/explain \
  -H 'Content-Type: application/json' \
  -d '{
        "code": "def add(a, b):\n    return a + b",
        "language": "python",
        "question": "Explain what this does and potential edge cases.",
        "top_k": 4
      }' | jq
```

## Environment Variables
Create `.env` (see `.env.example`):
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (optional, default: `text-embedding-3-large`)
- `CHROMA_DB_DIR` (optional, default: `./storage/chroma`)
- `MAX_CONTEXT_DOCS` (optional, default: `4`)

## Project Structure
```
app/
  main.py
  config.py
  models/
    schemas.py
  rag/
    vectorstore.py
    indexer.py
  services/
    explainer.py
  routes/
    explain.py
scripts/
  ingest.py
storage/
  chroma/  # persisted vector store
```

## Notes
- The explainer works without any indexed data, but RAG improves accuracy by retrieving related snippets from your codebase.
- For security, the API does not execute code. It performs static analysis via LLM + retrieval.
- You can run ingestion multiple times; content is deduplicated by embeddings.