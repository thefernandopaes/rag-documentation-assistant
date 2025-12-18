### Title
DocRag – Technical Documentation Intelligence Platform

### Summary
A production-ready RAG (Retrieval-Augmented Generation) platform for technical documentation with specialized capabilities for API documentation, code examples, and developer resources. Built with Flask, ChromaDB, and OpenAI, featuring intelligent document processing, semantic search, and automated code generation.

**Specialized in API documentation but supports any technical documentation format** – from OpenAPI/Swagger specs to framework guides, cloud platform docs, and DevOps tools.

### Capabilities Overview
- **Multi-domain Documentation Support**: API docs, frameworks, cloud platforms, DevOps tools, libraries
- **Intelligent Discovery**: Automatic detection of OpenAPI/Swagger specs, API patterns, and documentation structures
- **Multi-format Processing**: OpenAPI/Swagger (JSON/YAML), HTML, Markdown, Postman collections (extensible)
- **Code Generation**: Multi-language examples (cURL, Python, JavaScript, Node.js, PHP) with authentication handling
- **Semantic Search**: Vector-based retrieval with ChromaDB for contextually relevant answers
- **Production-Ready**: Comprehensive security, monitoring, rate limiting, and error handling

### Use Cases & Domains

#### Primary: API Documentation (Specialized)
- **REST APIs**: OpenAPI/Swagger specs, endpoint analysis, parameter documentation
- **Authentication**: Bearer tokens, API keys, OAuth flows
- **Code Examples**: Language-specific implementation samples with proper auth
- **Pre-configured Sources**: Stripe, GitHub, OpenAI, Twilio, Discord APIs

#### Extended: Technical Documentation (Supported)
- **Frameworks & Libraries**: React, Django, FastAPI, Express.js, Spring Boot
- **Cloud Platforms**: AWS, Azure, GCP documentation and service guides
- **DevOps Tools**: Docker, Kubernetes, Terraform, CI/CD platforms
- **Programming Languages**: Python, JavaScript, Java, Go, Rust documentation
- **Developer Tools**: Git, npm, pip, build tools, package managers

#### Enterprise Applications
- **Internal Knowledge Bases**: Corporate policies, procedures, best practices
- **Technical Guides**: Architecture docs, deployment guides, runbooks
- **Training Materials**: Onboarding docs, tutorials, how-to guides
- **Support Documentation**: FAQ systems, troubleshooting guides

### Contents
- Overview & Architecture
- Technology Stack
- Repository Layout
- Quickstart (Local)
- Configuration (Environment Variables)
- Database Migrations (Alembic)
- Documentation Ingestion
- Deployment (Railway + Gunicorn + TLS)
- Operations Runbook
- Security & Secrets
- Engineering Practices
- Troubleshooting
- Pre-configured Sources
- Key Features
- Extensibility & Customization
- License

### Overview & Architecture

#### Core Components
- **Document Processor**: Multi-format parsing with intelligent chunking strategies
  - OpenAPI/Swagger spec parser for structured API documentation
  - HTML/Markdown extraction with semantic structure preservation
  - Extensible architecture for custom document formats

- **RAG Engine**: Vector-based retrieval with context-aware generation
  - ChromaDB for semantic similarity search (cosine distance)
  - OpenAI embeddings (text-embedding-3-small) for dense vector representations
  - GPT-4o for response generation with domain-specific prompts
  - Conversation history management and context preservation

- **Code Generator**: Template-based multi-language code examples
  - Support for 8+ languages (cURL, Python, JavaScript, Node.js, PHP, Ruby, Go, Java)
  - Automatic authentication header injection
  - Parameter substitution with type-aware defaults

- **Discovery Engine**: Automatic documentation source detection
  - OpenAPI/Swagger spec discovery (common paths, robots.txt, sitemaps)
  - API documentation pattern recognition
  - Confidence scoring for source validation

#### Architecture Principles
- **Separation of Concerns**: Modular design with clear component boundaries
- **Idempotency**: Content hashing prevents duplicate processing
- **Security by Default**: Request validation, SSRF protection, rate limiting
- **Observability**: Comprehensive logging, monitoring, and analytics
- **Extensibility**: Plugin-friendly architecture for custom processors

### Technology Stack

#### Backend & API
- **Python 3.11+**: Core language with type hints and modern features
- **Flask 3.x**: Lightweight web framework with application factory pattern
- **SQLAlchemy 2.x**: ORM for metadata and analytics storage
- **Gunicorn**: Production WSGI server with worker management

#### RAG & AI
- **ChromaDB**: Vector database with persistent storage and cosine similarity
- **OpenAI API**:
  - text-embedding-3-small (1536-dim vectors)
  - GPT-4o for response generation
- **LangChain**: Text splitting utilities and document processing
- **Tiktoken**: Token counting for cost management

#### Data Processing
- **Trafilatura**: Web content extraction and HTML parsing
- **Beautiful Soup**: HTML parsing and structured data extraction
- **PyYAML**: YAML processing for OpenAPI specs
- **Requests**: HTTP client with retry logic

#### Storage & Caching
- **PostgreSQL**: Relational storage for metadata, analytics, conversations
- **SQLite**: Local development database option
- **File-based Cache**: TTL-based caching for API responses and specs

#### Security & Monitoring
- **Request Validation**: Input sanitization and SSRF protection
- **Rate Limiting**: Per-IP throttling with configurable limits
- **Monitoring**: Usage analytics, performance tracking, error reporting
- **Secrets Management**: Environment-based configuration with validation

#### Frontend
- **Jinja2**: Server-side templating
- **Bootstrap 5**: Responsive UI framework
- **Prism.js**: Syntax highlighting for code examples (JSON, YAML, HTTP, Bash, Python, JS)
- **Vanilla JavaScript**: Client-side interactivity

### Repository Layout
```
.
├── Core Application
│   ├── app.py                      # Flask application factory
│   ├── routes.py                   # API endpoints and request handlers
│   ├── models.py                   # SQLAlchemy database models
│   ├── config.py                   # Configuration management
│   └── main.py                     # Application entry point
│
├── RAG & AI Components
│   ├── rag_engine.py               # RAG engine (embedding, retrieval, generation)
│   ├── api_discovery.py            # Automatic API documentation discovery
│   ├── code_generator.py           # Multi-language code example generator
│   └── document_processor.py       # Multi-format document processing
│
├── Security & Operations
│   ├── security.py                 # Request validation and SSRF protection
│   ├── monitoring.py               # Analytics and performance tracking
│   ├── rate_limiter.py             # Request throttling
│   └── cache_manager.py            # TTL-based caching layer
│
├── Database & Migrations
│   ├── migrations/                 # Alembic database migrations
│   │   └── versions/               # Migration scripts
│   └── alembic.ini                 # Alembic configuration
│
├── Frontend
│   ├── templates/                  # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   └── chat.html
│   └── static/                     # CSS, JavaScript, assets
│       ├── css/style.css
│       └── js/chat.js
│
├── Utilities & Scripts
│   ├── scripts/
│   │   └── ingest.py              # Document ingestion script
│   └── data/
│       └── sample_docs.py         # Sample data for development
│
├── Documentation
│   ├── docs/                       # Comprehensive project documentation
│   │   ├── architecture.md
│   │   ├── deployment.md
│   │   ├── security-guidelines.md
│   │   └── [12+ more docs]
│   └── plan/                       # Implementation plans
│
└── Configuration
    ├── .gitignore                  # Git ignore patterns
    ├── pyproject.toml              # Python dependencies
    ├── gunicorn.conf.py            # Production server config
    ├── Procfile                    # Deployment configuration
    └── .env.example                # Environment variable template
```

### Quickstart (Local)

#### 1. Prerequisites
- Python 3.11 or higher
- OpenAI API key with credits
- PostgreSQL (optional, SQLite works for development)

#### 2. Setup Environment
**Create virtual environment:**
```bash
# Linux/macOS
python3.11 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Install dependencies:**
```bash
pip install alembic beautifulsoup4 chromadb email-validator flask flask-sqlalchemy \
    gunicorn langchain langchain-community langchain-openai numpy openai \
    psycopg2-binary python-dotenv requests sqlalchemy tiktoken trafilatura werkzeug
```

#### 3. Configure Environment
Create `.env` file in project root:
```bash
cp .env.example .env
```

**Minimum required configuration:**
```env
# OpenAI API (required)
OPENAI_API_KEY=sk-your-openai-api-key

# Session secrets (generate with: python -c "import secrets; print(secrets.token_hex(32))")
SESSION_SECRET=your-64-character-hex-secret
ADMIN_API_KEY=your-64-character-hex-admin-key

# Database (SQLite for local development)
DATABASE_URL=sqlite:///instance/docrag.db

# Vector Database
CHROMA_DB_PATH=./chroma_db

# RAG Configuration
CHUNK_SIZE=1200
MAX_RESPONSE_TOKENS=3000
TEMPERATURE=0.3

# Features
API_DISCOVERY_ENABLED=true
CODE_EXAMPLES_LANGUAGES=curl,python,javascript,nodejs,php
```

#### 4. Initialize Database
```bash
# Create instance directory
mkdir -p instance

# Run migrations
alembic upgrade head
```

#### 5. Start Application
```bash
python main.py
```

Application available at: **http://localhost:5000**

#### 6. Load Sample Documentation (Optional)
```bash
# Load pre-configured API documentation
python scripts/ingest.py --discover-apis

# Or load specific URL
python scripts/ingest.py --url https://docs.example.com
```

### Configuration (Environment Variables)

#### Core Settings
- `OPENAI_API_KEY` (required): OpenAI API key for embeddings and generation
- `SESSION_SECRET` (required in prod): Flask session encryption key (64 hex chars)
- `ADMIN_API_KEY` (required in prod): Protects administrative endpoints
- `DATABASE_URL`: Database connection string (default: sqlite:///instance/docrag.db)
- `CHROMA_DB_PATH`: Vector database storage path (default: ./chroma_db)

#### RAG Configuration
- `CHUNK_SIZE`: Document chunk size in characters (default: 1200)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_RESPONSE_TOKENS`: Maximum tokens in generated response (default: 3000)
- `TEMPERATURE`: LLM temperature for response generation (default: 0.3)

#### Feature Flags
- `API_DISCOVERY_ENABLED`: Enable automatic API discovery (default: true)
- `CODE_EXAMPLES_LANGUAGES`: Comma-separated list of code languages (default: curl,python,javascript,nodejs,php)
- `MONITORING_ENABLED`: Enable usage analytics (default: true)

#### Security
- `ALLOWED_DOMAINS`: Comma-separated list of allowed domains for crawling
- `REQUEST_SIZE_LIMIT`: Maximum request size (default: 16KB)
- `RATE_LIMIT_PER_MINUTE`: Requests per minute per IP (default: 15)

#### Performance
- `CACHE_TTL`: Cache time-to-live in seconds (default: 3600)
- `DOC_MAX_PAGES_PER_SOURCE`: Maximum pages to crawl per source (default: 100)
- `DOC_CRAWL_DELAY`: Delay between requests in seconds (default: 1.0)

#### Database Pooling (PostgreSQL)
- `DB_SSLMODE`: SSL mode (require, disable, verify-full)
- `DB_POOL_SIZE`: Connection pool size (default: 5)
- `DB_MAX_OVERFLOW`: Maximum overflow connections (default: 10)

### Database Migrations (Alembic)

#### Local Development
```bash
# Apply all migrations
alembic upgrade head

# Check current version
alembic current

# View migration history
alembic history
```

#### Production (Railway/Heroku)
- **Recommended**: Create a separate worker dyno/service that runs `alembic upgrade head` on startup
- **Alternative**: Run migrations manually via CLI before deploying new version
- **Last Resort**: Apply SQL migrations directly to database

### Documentation Ingestion

#### Automatic Discovery (Recommended)
Use built-in discovery for popular documentation sources:
```bash
# Discover and process pre-configured sources
API_DISCOVERY_ENABLED=true python scripts/ingest.py --discover-apis
```

#### Manual Ingestion
**Process any URL:**
```bash
python scripts/ingest.py --url https://docs.example.com
```

**Process OpenAPI/Swagger spec:**
```bash
python scripts/ingest.py --openapi-spec https://api.example.com/openapi.json
```

**Custom configuration:**
```bash
python scripts/ingest.py --url https://example.com/docs \
    --max-pages 50 \
    --crawl-delay 1.5
```

#### Sample Data (Development)
```bash
python scripts/ingest.py --use-sample
```

#### Supported Formats
- **OpenAPI/Swagger**: JSON and YAML specifications
- **HTML**: Documentation websites with semantic markup
- **Markdown**: Technical guides and tutorials
- **Postman Collections**: API collection exports (extensible)

#### Features
- **Idempotency**: Content hashing prevents duplicate processing
- **Incremental Updates**: Only processes changed documents
- **Automatic Caching**: Spec caching with 24h TTL
- **Rate Limiting**: Respectful crawling with configurable delays

### Deployment (Railway + Gunicorn + TLS)

#### Railway Configuration
**Procfile:**
```
web: gunicorn -c gunicorn.conf.py app:app
```

**Environment Variables:**
Set all required variables in Railway dashboard (same as local .env)

**Persistent Volume:**
- Mount volume at `/var/lib/docrag`
- Set `CHROMA_DB_PATH=/var/lib/docrag/chroma_db`
- Ensures vector database persists across deployments

**Worker Process (Optional):**
For automatic documentation updates:
```bash
API_DISCOVERY_ENABLED=true python scripts/ingest.py --discover-apis
```

#### Production Checklist
- [ ] Set `ENV=production` or `FLASK_ENV=production`
- [ ] Generate secure `SESSION_SECRET` and `ADMIN_API_KEY` (64 hex chars)
- [ ] Configure PostgreSQL database with SSL (`DB_SSLMODE=require`)
- [ ] Set up persistent volume for ChromaDB
- [ ] Enable monitoring (`MONITORING_ENABLED=true`)
- [ ] Configure allowed domains (`ALLOWED_DOMAINS`)
- [ ] Set appropriate rate limits
- [ ] Enable HTTPS/TLS termination
- [ ] Configure database connection pooling
- [ ] Set up log aggregation
- [ ] Configure backup strategy for PostgreSQL and ChromaDB

### Operations Runbook

#### Health Checks
- **Application Health**: `GET /healthz` - Returns DB status and vector count
- **System Stats**: `GET /api/stats` - Detailed analytics and metrics

#### Monitoring
- **Usage Analytics**: Track queries, response times, feedback
- **Performance Metrics**: Cache hit rates, retrieval latency, generation time
- **Error Tracking**: Failed requests, validation errors, LLM failures
- **System Health**: Database connections, vector store status

#### Scaling Considerations
- **Horizontal Scaling**: Stateless design supports multiple instances
- **Database Pooling**: Configure `DB_POOL_SIZE` and `DB_MAX_OVERFLOW`
- **Worker Concurrency**: Adjust `WEB_CONCURRENCY` for Gunicorn
- **Cache Strategy**: File-based cache with TTL for spec caching
- **Vector Store**: ChromaDB supports concurrent reads

#### Backup & Recovery
- **PostgreSQL**: Regular automated backups with point-in-time recovery
- **ChromaDB**: Backup persistent volume, can rebuild from source documents
- **Configuration**: Version control all config, document environment variables

### Security & Secrets

#### Secret Management
- **Never commit** `.env` files or real secrets to version control
- Use `.env.example` with placeholder values for documentation
- Rotate credentials if exposed or on a regular schedule
- Use environment variables or secret management services in production

#### Security Features
- **Request Validation**: Input sanitization, size limits, type checking
- **SSRF Protection**: URL validation, domain allowlists, redirect following limits
- **Rate Limiting**: Per-IP throttling to prevent abuse
- **SQL Injection**: Protected by SQLAlchemy ORM and parameterized queries
- **XSS Prevention**: Template auto-escaping, output sanitization
- **CSRF**: Session-based protection for state-changing operations

#### Production Security
- `ADMIN_API_KEY` protects administrative endpoints (`/api/initialize`)
- `DB_SSLMODE=require` enforces encrypted database connections
- `SESSION_COOKIE_SECURE` enforces HTTPS-only cookies
- `SESSION_COOKIE_HTTPONLY` prevents JavaScript access to session
- `SESSION_COOKIE_SAMESITE=Lax` prevents CSRF attacks

#### Security Headers (via ProxyFix)
- Proper scheme/host/port/IP detection behind TLS-terminating proxies
- Secure cookie settings in production
- Content Security Policy ready

### Engineering Practices

#### Code Organization
- **Small, cohesive modules**: Files typically < 300 lines
- **Intention-revealing names**: Clear, descriptive identifiers
- **Shallow nesting**: Guard clauses and early returns
- **Type hints**: Comprehensive typing for better tooling and documentation

#### Code Quality
- **Comments explain "why"**, not "what"
- **No duplicated logic**: DRY principle, centralized config/validation
- **Error handling**: Graceful degradation, user-friendly messages
- **Logging**: Structured logging with appropriate levels

#### Development Workflow
- **Small, focused PRs**: Easier review and reduced risk
- **Feature branches**: Isolate changes, enable parallel development
- **Code review**: Mandatory review before merge
- **Continuous Integration**: Automated linting, formatting, testing (recommended)

#### Testing Strategy (Recommended)
- **Unit tests**: Document processing, validators, utilities
- **Integration tests**: RAG pipeline, API endpoints, database
- **Smoke tests**: Ingestion pipeline, health checks
- **Regression tests**: Prompt evaluation, answer quality (future enhancement)

### Troubleshooting

#### Common Issues

**API Discovery Problems:**
- Check `ALLOWED_DOMAINS` configuration
- Verify URL accessibility from server
- Review discovery logs for HTTP errors
- Confirm OpenAPI spec is valid JSON/YAML

**Code Generation Issues:**
- Verify `CODE_EXAMPLES_LANGUAGES` config
- Check OpenAPI spec format and completeness
- Review endpoint definitions for required parameters

**Empty Search Results:**
- Confirm documentation ingestion completed
- Check vector store count: `GET /api/stats`
- Verify ChromaDB path and permissions
- Review chunking strategy for domain

**Performance Problems:**
- Monitor cache hit rates in analytics
- Check OpenAI API latency
- Optimize chunk size for domain
- Review database query performance

**Database Errors:**
- `UndefinedColumn content_hash`: Run `alembic upgrade head`
- Connection issues: Check `DATABASE_URL` and network
- Migration failures: Review migration logs, apply manually if needed

**Deployment Issues:**
- Not suitable for Vercel (requires persistent storage)
- Use Railway, Heroku, AWS Elastic Beanstalk, or similar
- Ensure persistent volumes for ChromaDB and SQLite (if used)

### Pre-configured Documentation Sources

#### API Documentation (Specialized)
- **Stripe**: Payment processing API - OpenAPI spec + comprehensive docs
- **GitHub**: Repository management API - REST API with detailed examples
- **OpenAI**: AI/ML services API - GPT, embeddings, fine-tuning
- **Twilio**: Communication services API - Voice, SMS, messaging
- **Discord**: Bot development API - Gateway, interactions, webhooks

#### Extensibility
Additional sources can be configured in `config.py`:
```python
DOC_SOURCES = {
    "your-docs": {
        "base_url": "https://docs.yoursite.com/",
        "docs_url": "https://docs.yoursite.com/guide/",
        "type": "technical"  # or "api", "framework", "cloud", etc.
    }
}
```

**Supported documentation types:**
- REST APIs with OpenAPI/Swagger specs
- Framework and library documentation
- Cloud platform service guides
- DevOps tool documentation
- Programming language references
- Technical tutorials and guides

### Key Features

#### Intelligent Document Processing
- **Multi-format Support**: OpenAPI, HTML, Markdown, Postman collections
- **Semantic Chunking**: Context-aware splitting preserving structure
- **Automatic Discovery**: Pattern-based detection of API documentation
- **Idempotent Processing**: Content hashing prevents duplicates

#### Advanced RAG Pipeline
- **Vector Search**: Semantic similarity with ChromaDB (cosine distance)
- **Contextual Retrieval**: Top-k relevant chunks with relevance scoring
- **Conversation Memory**: Multi-turn dialogue with history preservation
- **Domain-Specific Prompts**: Specialized prompts for technical documentation

#### Code Generation
- **Multi-language Support**: 8+ programming languages
- **Authentication Handling**: Automatic auth header injection (Bearer, API Key, Basic)
- **Parameter Substitution**: Type-aware default values
- **Format Preservation**: Maintains API structure and relationships

#### Production Features
- **Comprehensive Monitoring**: Usage analytics, performance tracking, error reporting
- **Security by Default**: SSRF protection, rate limiting, input validation
- **Caching Strategy**: TTL-based caching for specs and responses
- **Observability**: Structured logging, health checks, metrics endpoints
- **Scalability**: Stateless design, database pooling, horizontal scaling ready

#### User Experience
- **Real-time Chat**: Interactive Q&A interface with typing indicators
- **Syntax Highlighting**: Prism.js for code examples (8+ languages)
- **Structured Display**: HTTP method badges, endpoint formatting, parameter tables
- **Related Concepts**: Contextual suggestions for exploration

### Extensibility & Customization

#### Adding New Documentation Sources
1. Configure source in `config.py`:
```python
DOC_SOURCES["new-docs"] = {
    "base_url": "https://docs.example.com/",
    "docs_url": "https://docs.example.com/guide/",
    "type": "framework"
}
```

2. Run ingestion:
```bash
python scripts/ingest.py --url https://docs.example.com
```

#### Custom Document Processors
Extend `DocumentProcessor` class for specialized formats:
```python
class CustomProcessor(DocumentProcessor):
    def process_custom_format(self, content: str) -> List[Document]:
        # Custom processing logic
        return documents
```

#### Custom Chunking Strategies
Implement domain-specific chunking in `document_processor.py`:
```python
def custom_chunk_strategy(self, content: str) -> List[str]:
    # Domain-specific chunking logic
    return chunks
```

#### Prompt Customization
Modify prompts in `rag_engine.py` for domain-specific behavior:
```python
CUSTOM_SYSTEM_PROMPT = """
You are an expert in [your domain].
Provide clear, accurate answers based on the documentation.
"""
```

### Migration from API-Only Version
This platform evolved from a specialized API documentation assistant. See:
- **Complete Migration Guide**: `docs/API_MIGRATION_COMPLETE.md`
- **Implementation Plan**: `plan/api-rag-migration-plan.md`
- **Backward Compatibility**: All API-specific features maintained
- **Enhanced Capabilities**: 10x broader documentation support

### License
Reference implementation for production-grade technical documentation RAG platform. Adapt to your organization's policies and requirements.

---

**Built for portfolio demonstration** – showcases RAG implementation, LLM integration, document processing, and production engineering practices.
