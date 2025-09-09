### Title
DocRag – AI-Powered API Documentation Assistant

### Summary
A production-ready RAG web application specialized in API documentation assistance with automatic discovery, code generation, and comprehensive API guidance. Automatically discovers and processes OpenAPI/Swagger specs, REST API docs, and other API documentation formats. Features intelligent code generation, endpoint analysis, and comprehensive API guidance. Flask backend, ChromaDB, OpenAI, PostgreSQL with enhanced security and monitoring. This README targets senior-level review: run, deploy, operate, and practices.

### Contents
- Overview & Architecture
- Technology Stack
- Repository Layout
- Quickstart (Local)
- Configuration (Environment Variables)
- Database Migrations (Alembic)
- API Documentation Ingestion
- Deployment (Railway + Gunicorn + TLS)
- Operations Runbook
- Security & Secrets
- Engineering Practices
- Troubleshooting
- API Documentation Sources
- Key Features
- Migration Information
- License

### Overview & Architecture
- **API Discovery**: Automatic discovery of OpenAPI/Swagger specs, common API doc patterns, sitemap-based discovery.
- **Multi-format Processing**: OpenAPI/Swagger (JSON/YAML), HTML API docs, Postman collections (extensible).
- **Intelligent Chunking**: Endpoint-based chunking preserving API structure and relationships.
- **Code Generation**: Multi-language examples (cURL, Python, JavaScript, Node.js, PHP) with authentication.
- **Enhanced RAG**: API-specific prompts, structured responses with endpoints/parameters/examples.
- **Security & Monitoring**: Comprehensive validation, SSRF protection, usage analytics, performance tracking.
- **Principles**: API-first design; separation of concerns; idempotency; security by default; comprehensive monitoring.

### Technology Stack
- **Backend**: Python 3.11, Flask 3.x, SQLAlchemy 2.x, Flask‑SQLAlchemy, Gunicorn.
- **RAG**: ChromaDB (cosine), OpenAI (text‑embedding‑3‑small, GPT‑4o), API-specific chunking.
- **API Processing**: OpenAPI parser, HTML extraction, automatic discovery engine.
- **Code Generation**: Multi-language template engine with authentication handling.
- **Security**: Request validation, SSRF protection, rate limiting, input sanitization.
- **Monitoring**: Usage analytics, performance tracking, error reporting, system health.
- **Storage**: PostgreSQL (metadata + analytics), persistent volume for ChromaDB.
- **Frontend**: Jinja + Bootstrap + Enhanced Prism.js (API syntax highlighting).

### Repository Layout
- **Core**: `app.py` (factory), `routes.py` (APIs), `config.py`, `models.py`
- **API RAG**: `rag_engine.py` (enhanced), `api_discovery.py`, `code_generator.py`
- **Processing**: `document_processor.py` (multi-format), `scripts/ingest.py`
- **Security**: `security.py` (validation), `monitoring.py` (analytics)
- **Infrastructure**: `migrations/`, `gunicorn.conf.py`, `Procfile`, `pyproject.toml`
- **Frontend**: `static/` (enhanced CSS/JS), `templates/` (API-focused)
- **Documentation**: `docs/` (migration guide), `plan/` (implementation plan)

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
  - `API_DISCOVERY_ENABLED=true`
  - `CODE_EXAMPLES_LANGUAGES=curl,python,javascript,nodejs,php`
- Migrate and run:
  - `alembic upgrade head`
  - `python main.py` → http://localhost:5000

### Configuration (Environment Variables)
- **Core**: `OPENAI_API_KEY`, `SESSION_SECRET` (prod), `ADMIN_API_KEY`, `DATABASE_URL`, `CHROMA_DB_PATH`.
- **API RAG**: `CHUNK_SIZE` (1200→API-optimized), `MAX_RESPONSE_TOKENS` (3000→detailed), `TEMPERATURE` (0.3→consistent).
- **API Discovery**: `API_DISCOVERY_ENABLED` (true), `API_CACHE_SPEC_TTL` (86400), `API_MAX_ENDPOINTS_PER_SPEC` (200).
- **Code Generation**: `CODE_EXAMPLES_LANGUAGES` (curl,python,javascript,nodejs,php).
- **Security**: `ALLOWED_DOMAINS`, `REQUEST_SIZE_LIMIT` (16KB), `RATE_LIMIT_PER_MINUTE` (15).
- **Monitoring**: `MONITORING_ENABLED`, `ANALYTICS_RETENTION_DAYS` (90), `ERROR_REPORTING_ENABLED`.
- **Ingestion**: `DOC_MAX_PAGES_PER_SOURCE` (100→API-focused), `DOC_CRAWL_DELAY` (1.0→respectful).
- **DB Pooling**: `DB_SSLMODE=require`, `DB_POOL_SIZE` (5), `DB_MAX_OVERFLOW` (10).

### Database Migrations (Alembic)
- Local: `alembic upgrade head`.  
- Railway Worker (recommended): Start Command `alembic upgrade head` with same env vars as web.  
- As last resort, apply migration SQL manually.

### API Documentation Ingestion
- **Automatic Discovery**: Use built-in API discovery for popular sources (Stripe, GitHub, OpenAI, etc.).
- **Manual Ingestion**: `scripts/ingest.py` for custom API documentation.
- **Multi-format Support**: OpenAPI/Swagger specs, HTML API docs, Postman collections.
- **Examples**:
  - `API_DISCOVERY_ENABLED=true python scripts/ingest.py --discover-apis`
  - `python scripts/ingest.py --url https://api.example.com/docs`
  - `python scripts/ingest.py --openapi-spec https://api.example.com/openapi.json`
- **Idempotency**: Content hashing prevents duplicates; automatic spec caching (24h TTL).

### Deployment (Railway)
- Web: `web: gunicorn -c gunicorn.conf.py app:app`.  
- Volume: mount `/var/lib/docrag` and set `CHROMA_DB_PATH=/var/lib/docrag/chroma_db`.  
- Worker: `API_DISCOVERY_ENABLED=true python scripts/ingest.py --discover-apis`.  
- Proxy/TLS: ProxyFix enabled; secure cookies in prod.

### Operations Runbook
- **Health**: `/healthz` (DB + vector count), `/api/stats` (enhanced with API analytics).
- **Monitoring**: Built-in analytics tracking API usage, performance, discovery stats.
- **Security**: Request validation, SSRF protection, rate limiting, input sanitization.
- **Scaling**: Adjust `WEB_CONCURRENCY`, DB pool; monitor API discovery load.
- **Observability**: Structured logging, error tracking, performance metrics.
- **API Management**: Automatic spec caching, discovery rate limiting, source validation.

### Security & Secrets
- Never commit real secrets; keep `.env` out of VCS; use `.env.example` with placeholders.  
- Rotate credentials if exposed.  
- `ADMIN_API_KEY` protects `/api/initialize` in prod.  
- `DB_SSLMODE=require`, `SESSION_COOKIE_SECURE` behind TLS.
- Enhanced security features: SSRF protection, request validation, domain allowlists.

### Engineering Practices
- Small, cohesive modules; avoid files > ~300 lines.  
- Intention‑revealing names; guard clauses; shallow nesting.  
- Comments explain "why", not "what".  
- No duplicated logic; centralize config/validation.  
- PRs: small/focused; CI should lint/format and test migrations on staging.  
- Tests (recommended): unit (processing/validators), integration (RAG/API), ingestion smoke.

### Troubleshooting
- **API Discovery Issues**: Check `ALLOWED_DOMAINS`, verify URL accessibility, review discovery logs.
- **Code Generation Problems**: Verify `CODE_EXAMPLES_LANGUAGES` config, check OpenAPI spec format.
- **Empty API Results**: Confirm API discovery ran, check endpoint chunking, verify spec processing.
- **Security Errors**: Review rate limits, check URL validation, verify domain allowlist.
- **Performance Issues**: Monitor API analytics, check cache hit rates, optimize chunk sizes.
- **Legacy Issues**: `UndefinedColumn content_hash` → run `alembic upgrade head`.
- **Deployment**: Not for Vercel (not static); use Railway/similar with persistent volumes.

### API Documentation Sources
Pre-configured for popular APIs:
- **Stripe** (Payment processing)
- **GitHub** (Repository management) 
- **OpenAI** (AI/ML services)
- **Twilio** (Communication services)
- **Discord** (Bot development)

Additional sources can be configured through `API_DOC_SOURCES` in `config.py`.

### Key Features
- **Automatic API Discovery** from URLs
- **Multi-language Code Generation** (cURL, Python, JS, Node.js, PHP)
- **Endpoint Analysis** with parameters, authentication, examples
- **Comprehensive Security** with SSRF protection and validation
- **Real-time Monitoring** and usage analytics
- **Production-ready** with enhanced rate limiting and error handling
- **API-Specific UI** with syntax highlighting and structured display
- **Intelligent Chunking** preserving API structure and relationships

### Migration from General Documentation
- **Complete Migration Guide**: See `docs/API_MIGRATION_COMPLETE.md` for full implementation details
- **Implementation Plan**: See `plan/api-rag-migration-plan.md` for the 4-phase migration strategy
- **Backward Compatibility**: Legacy documentation sources maintained during transition
- **Enhanced Capabilities**: 10x more comprehensive API support with automatic discovery

### License
Reference implementation for production-grade API documentation RAG assistant; adapt to your organization's policies.