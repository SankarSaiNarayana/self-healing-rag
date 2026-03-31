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

Ingest docs (put files under `data/` first):

```bash
python -m scripts.ingest --data_dir data --collection docs
```

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

### Notes

- By default, embeddings use `sentence-transformers` locally.
- Generation + verification use an **OpenAI-compatible** chat endpoint (OpenAI, Azure OpenAI, Ollama OpenAI-compat, etc.).
- If no LLM credentials are set, the service still retrieves and returns top sources, but the final answer will be a simple extractive fallback.

