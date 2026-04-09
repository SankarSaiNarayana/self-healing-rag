## Self-healing RAG service

Implements the pipeline in your diagram:

- query classification (factual / conceptual / multi-hop)
- hybrid retrieval (dense vector + BM25) + optional re-rank
- retrieval quality check (may re-query)
- answer generation with retrieved context
- hallucination / claim check (may re-retrieve + re-generate)
- citation verification (attach source for each claim)
- returns verified answer + sources + confidence

### Quickstart (local)

Create env file:

```bash
cp .env.example .env
```

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the sample PDF (optional) and ingest docs under `data/`:

```bash
python scripts/build_sample_pdf.py --out data/demo.pdf
python -m scripts.ingest --data_dir data --collection docs
```

Supported file types include `.txt`, `.md`, and `.pdf` (PDF text is extracted with `pypdf`).

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Query:

```bash
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is this system?","collection":"docs"}' | jq .
```

### Quickstart (Docker)

```bash
docker compose up --build
```

Then query `http://localhost:8000/query`.

### Using Qdrant (optional)

To use Qdrant as the vector database:

- Set `VECTOR_DB=qdrant` in `.env`
- Ensure Qdrant is running (the provided `docker-compose.yml` starts it on `http://localhost:6333`)

### Notes

- By default, embeddings use `sentence-transformers` locally.
- Generation + verification use an **OpenAI-compatible** chat endpoint (OpenAI, Azure OpenAI, Ollama OpenAI-compat, etc.).
- If no LLM credentials are set, the service still retrieves and returns top sources, but the final answer will be a simple extractive fallback.

### Web demo UI

Open `http://localhost:8000/demo` after starting the API (served from `static/index.html`).

### Debug: memory inspection

```bash
curl -s http://localhost:8000/memory/demo
```

If `API_KEY` is set in `.env`, pass `X-API-Key: <key>`.

### Production-oriented features (built-in)

- **Optional API key**: set `API_KEY` in `.env`; clients send `X-API-Key`.
- **Rate limiting**: `RATE_LIMIT` (e.g. `120/minute`) via SlowAPI.
- **CORS**: `CORS_ORIGINS` (comma-separated origins, or `*`).
- **Request IDs**: `X-Request-ID` on responses; structured request logging.
- **Health**: `GET /health` includes vector store ping; set `HEALTH_CHECK_LLM=true` to probe the LLM (uses quota).
- **Warnings**: `POST /query` returns `warnings` when the LLM is skipped or errors (extractive fallback).
- **Episodic memory**: deduplicated “what did I ask before” answers with timestamps.
- **PDF ingest**: drop `.pdf` files under `data/` and run `scripts.ingest` (uses `pypdf`).
- **CI**: GitHub Actions workflow runs import + `pytest`.

### Eval script (quick)

With the API running:

```bash
python scripts/eval_golden.py --base http://127.0.0.1:8000
```

