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

Create env file at the **project root** (same folder as `README.md`):

```bash
cp .env.example .env
```

The app loads **`/path/to/rag/.env`** automatically even if you start `uvicorn` from another directory. For real answers (not extractive fallback), set **`OPENAI_BASE_URL`**, **`OPENAI_API_KEY`**, and **`OPENAI_MODEL`** (any OpenAI-compatible provider). Check `GET /health`: `llm_enabled` should be `true`, and `llm` shows which vars are missing.

**Groq (free tier):** keys and models are managed in the [Groq console](https://console.groq.com/). Use `OPENAI_BASE_URL=https://api.groq.com/openai/v1`, put your Groq secret in **`OPENAI_API_KEY`** (variable name stays `OPENAI_*` because the HTTP client is OpenAI-compatible), and set **`OPENAI_MODEL`** to a Groq model id (see `.env.example`).

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

### Upload documents (API + demo UI)

Users can index their own files into a collection, then query that collection.

- **Demo**: open `http://localhost:8000/demo`, pick files under **Upload documents**, then run a question (same **Collection** name).
- **API** (multipart field name must be `files`; repeat the field for multiple files):

```bash
curl -s -X POST "http://localhost:8000/collections/docs/documents" \
  -H "X-API-Key: $API_KEY" \
  -F "files=@./README.md" \
  -F "files=@./notes.txt"
```

Tune limits via `UPLOAD_MAX_FILES`, `UPLOAD_MAX_BYTES_PER_FILE`, `UPLOAD_MAX_TOTAL_BYTES`, `UPLOAD_RATE_LIMIT` in `.env`.

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

Open `http://localhost:8000/demo` after starting the API (served from `static/index.html`). The page includes **presenter notes** (expand “How to showcase self-healing”) and a **Self-healing trace** panel after each query (`retries`, claim counts, warnings) so a live audience can see the two loops without reading raw JSON.

### Live demo script (talk track)

- **Clarify the “database”**: the **vector index** (Chroma locally, or Qdrant if configured) stores **chunks + embeddings** for your files. It does not “heal” by itself. **Self-healing** is **query-time behavior**: optional extra retrieval and optional re-generation when quality checks fail.
- **Show loop 1**: run a question, point at **`retries.retrieval`** in the trace (or JSON). If `> 0`, say: “The first search looked weak, so the pipeline widened retrieval and tried again.”
- **Show loop 2**: point at **`retries.verification`** and **claim** chips. If verification `> 0`, say: “Several sentences were not well supported by sources, so it pulled more context and asked the model again—bounded retries.”
- **Show grounding**: expand **`claims`** in the JSON: each item is a sentence checked against a **source chunk** (or flagged). That is the “not making things up” story.
- **Optional** (to make retries more likely during a rehearsed demo): temporarily lower **`MIN_RETRIEVAL_SCORE`** in `.env` (e.g. `0.45`) so marginal matches trigger more retrieval retries; restore afterward.

### Deploy

- **Docker image**: the included `Dockerfile` listens on `$PORT` (for **Render**, **Fly.io**, **Railway**, etc.).
- **Render**: connect the GitHub repo, choose “Docker”, set root `render.yaml` or point the Dockerfile, then add environment variables (`OPENAI_*`, optional `API_KEY`, `VECTOR_DB`, `QDRANT_*`). Default **Chroma** on a web dyno is **ephemeral** unless you attach a [Render Disk](https://render.com/docs/disks) and set `CHROMA_DIR` to the mount path—or use **Qdrant Cloud** (`VECTOR_DB=qdrant`).
- **Fly.io**: install the [Fly CLI](https://fly.io/docs/hands-on/install-flyctl/), run `fly launch` in this directory (uses `fly.toml`), then `fly secrets set` for `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and optional `API_KEY`. The sample `fly.toml` uses `internal_port = 8080` to match Fly’s `PORT`.
- **Secrets**: never commit `.env`; set LLM and DB keys only in the host’s secret store.

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

