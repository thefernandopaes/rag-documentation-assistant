### Title
DocRag – AI-Powered Technical Documentation Assistant

### Summary
A production-ready RAG web application answering questions from official documentation (React, Python, FastAPI). Flask backend, ChromaDB, OpenAI, Postgres for metadata, rate limiting and chat history. This README targets senior-level review: run, deploy, operate, and practices.

### Contents
- Overview & Architecture
- Technology Stack
- Repository Layout
- Quickstart (Local)
- Configuration (Environment Variables)
- Database Migrations (Alembic)
- Ingestion (Offline Job – Crawl & Upsert)
- Deployment (Railway + Gunicorn + TLS)
- Operations Runbook
- Security & Secrets
- Engineering Practices
- Troubleshooting
- License

### Overview & Architecture
- Ingestion (offline): crawl docs, extract, split (LangChain), embed (OpenAI), upsert to Chroma; store metadata in Postgres.  
- Retrieval (online): embed query → retrieve top-k from Chroma.  
- Generation (online): build compact context → GPT‑4o → JSON with answer, examples, sources.  
- Principles: separation of concerns; idempotency (content hashes); caching/rate limiting/timeouts/retries; env‑driven config.

### Technology Stack
- Backend: Python 3.11, Flask 3.x, SQLAlchemy 2.x, Flask‑SQLAlchemy, Gunicorn.  
- RAG: ChromaDB (cosine), OpenAI (text‑embedding‑3‑small, GPT‑4o), LangChain splitters.  
- Storage: Postgres (metadata), persistent volume for Chroma.  
- Ops: Alembic; retries/backoff; health endpoints.  
- Frontend: Jinja + Bootstrap + Prism.js.

### Repository Layout
- `app.py` (app factory), `routes.py` (APIs), `rag_engine.py` (RAG), `document_processor.py` (crawler), `models.py`, `config.py`, `scripts/ingest.py`, `migrations/`, `static/`, `templates/`, `gunicorn.conf.py`, `Procfile`, `pyproject.toml`.

### Quickstart (Local)
- Create venv and install deps:
  - `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`)
  - `pip install -r <(python - <<'PY'\nimport tomllib;print('\\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))\nPY\n)`
- `.env` (never commit):
  - `OPENAI_API_KEY=your-openai-key`
  - `SESSION_SECRET=your-64-hex`
  - `ADMIN_API_KEY=your-64-hex`
  - `DATABASE_URL=postgresql://user:password@localhost:5432/docrag`
  - `CHROMA_DB_PATH=./chroma_db`
- Migrate and run:
  - `alembic upgrade head`
  - `python main.py` → http://localhost:5000

### Configuration (Environment Variables)
- Core: `OPENAI_API_KEY`, `SESSION_SECRET` (prod), `ADMIN_API_KEY`, `DATABASE_URL`, `CHROMA_DB_PATH`.  
- RAG: `CHUNK_SIZE` (1000→recommend 800), `CHUNK_OVERLAP` (200→150), `MAX_RESPONSE_TOKENS` (2000→700), `TEMPERATURE` (0.7→0.25).  
- Ingestion: `DOC_USE_SAMPLE`, `DOC_MAX_PAGES_PER_SOURCE` (300–500), `DOC_CRAWL_DELAY` (0.5–1.0), `DOC_CRAWL_TIMEOUT` (15).  
- Resilience: `OPENAI_TIMEOUT` (30), `OPENAI_MAX_RETRIES`(2), `REQUESTS_MAX_RETRIES`(2), `REQUESTS_BACKOFF_FACTOR`(0.5).  
- DB Pooling: `DB_SSLMODE=require`, `DB_POOL_SIZE` (5), `DB_MAX_OVERFLOW` (10).

### Database Migrations (Alembic)
- Local: `alembic upgrade head`.  
- Railway Worker (recommended): Start Command `alembic upgrade head` with same env vars as web.  
- As last resort, apply migration SQL manually.

### Ingestion (Offline Job – Crawl & Upsert)
- Never crawl on request path. Use `scripts/ingest.py`.  
- Examples:
  - `DOC_USE_SAMPLE=false python scripts/ingest.py --max-pages 300`
  - `python scripts/ingest.py --use-sample`  
- Idempotency: `content_hash` per chunk; upsert deletes old vectors for same source before insertion.

### Deployment (Railway)
- Web: `web: gunicorn -c gunicorn.conf.py app:app`.  
- Volume: mount `/var/lib/docrag` and set `CHROMA_DB_PATH=/var/lib/docrag/chroma_db`.  
- Worker: `DOC_USE_SAMPLE=false python scripts/ingest.py --max-pages 300`.  
- Proxy/TLS: ProxyFix enabled; secure cookies in prod.

### Operations Runbook
- Health: `/healthz` (DB + vector count), `/api/stats` (docs, cache, conversations).  
- Scaling: adjust `WEB_CONCURRENCY` and DB pool.  
- Observability: logs to stdout/stderr; add Sentry if needed.  
- Reliability: timeouts/retries/backoff; rate limiting; file cache.

### Security & Secrets
- Never commit real secrets; keep `.env` out of VCS; use `.env.example` with placeholders.  
- Rotate credentials if exposed.  
- `ADMIN_API_KEY` protects `/api/initialize` in prod.  
- `DB_SSLMODE=require`, `SESSION_COOKIE_SECURE` behind TLS.

### Engineering Practices
- Small, cohesive modules; avoid files > ~300 lines.  
- Intention‑revealing names; guard clauses; shallow nesting.  
- Comments explain “why”, not “what”.  
- No duplicated logic; centralize config/validation.  
- PRs: small/focused; CI should lint/format and test migrations on staging.  
- Tests (recommended): unit (processing/validators), integration (RAG/API), ingestion smoke.

### Troubleshooting
- Vercel 404: not a static app; deploy on Railway or set rewrites.  
- `UndefinedColumn content_hash`: run `alembic upgrade head`.  
- `postgres.railway.internal` unresolved locally: `railway connect` or run migrations via Worker.  
- Answers too long: lower `MAX_RESPONSE_TOKENS` and `TEMPERATURE`; reduce `CHUNK_SIZE` and re‑ingest.  
- Empty results: confirm ingestion ran; `CHROMA_DB_PATH` points to a persistent volume.

### License
- Reference implementation for a production-grade RAG assistant; adapt to your org’s policies.