# DocRag – Technical Documentation Intelligence Platform

A production-ready RAG (Retrieval-Augmented Generation) platform for technical documentation, specializing in API documentation discovery and code example generation. Built with **FastAPI** (Async), **Google Gemini**, and **ChromaDB**.

> **Note:** This project has been migrated from Flask/OpenAI to a high-performance **FastAPI/Gemini** architecture.

## 🚀 Key Features

*   **⚡ High-Performance Architecture**: Fully async implementation using FastAPI, `httpx`, and `asyncpg`.
*   **🧠 Advanced RAG Engine**: Hybrid search using **Google Gemini Embeddings** and **ChromaDB**.
*   **🤖 Intelligent Discovery**: Automatically crawls and indexes documentation (Sitemaps, OpenAPI/Swagger, HTML).
*   **📝 Code Generation**: Uses **Gemini 1.5 Flash** to generate accurate code examples in 8+ languages (Python, Node.js, cURL, etc.).
*   **🛡️ Production Ready**: Features rate limiting, security headers, JSON logging, and Docker/Railway compatibility.

## 🏗️ Technology Stack

*   **Backend**: Python 3.11+, **FastAPI** (Async)
*   **AI/LLM**: Google Gemini (`gemini-1.5-flash`), `google-genai` SDK
*   **Vector DB**: ChromaDB (Persistent)
*   **Database**: PostgreSQL (Metadata & History) via `SQLAlchemy 2.0` (Async)
*   **Processing**: `BeautifulSoup4`, `Trafilatura` (Content Extraction)

## 📂 Repository Layout

```text
.
├── app/
│   ├── api/                # FastAPI Routes & Dependencies
│   ├── core/               # RAG Engine & Logging
│   ├── db/                 # Database Models & Connections
│   ├── services/           # Document Processor & Crawler
│   └── utils/              # Validators & Helpers
├── scripts/
│   ├── ingest.py           # Document Ingestion Script
│   ├── reset_chroma.py     # Utility to clear Vector DB
│   └── fix_collation.sql   # Maintenance scripts
├── config.py               # Central Configuration
├── fastapi_app.py          # Application Entrypoint
├── Procfile                # Railway Deployment Config
└── pyproject.toml          # Dependencies
```

## 🛠️ Quickstart (Local Development)

### 1. Prerequisites
*   Python 3.11+
*   Google Gemini API Key
*   PostgreSQL (Optional, defaults to SQLite for local)

### 2. Setup
```bash
# Clone and enter directory
git clone <repo-url>
cd rag-documentation-assistant

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file based on `.env.example`:
```env
# AI Provider
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here

# Database (Default: SQLite)
DATABASE_URL=sqlite:///docrag.db

# Vector DB
CHROMA_DB_PATH=./chroma_db
```

### 4. Initialize & Run
```bash
# Initialize DB (Automatic on startup) and Start Server
uvicorn fastapi_app:app --reload
```
App will be ready at `http://localhost:8000`.

### 5. Ingest Documentation
To populate the vector database with documentation:
```bash
# Discover APIs automatically
API_DISCOVERY_ENABLED=true python scripts/ingest.py --discover-apis

# OR Ingest a specific URL
python scripts/ingest.py --url https://fastapi.tiangolo.com/
```

## 🔧 Maintenance

### Resetting Embeddings
If you face "dimension mismatch" errors or want to re-index:
1.  **Stop the running server** (Ctrl+C).
2.  Run the reset script:
    ```bash
    python scripts/reset_chroma.py
    ```
3.  Restart server and re-ingest.

### Database Migrations
We use Alembic for schema migrations (if using Postgres):
```bash
alembic upgrade head
```

## 🚀 Deployment (Railway)

The project is configured for **Railway** out of the box using `Procfile`.

1.  Push to GitHub.
2.  Connect repository to Railway.
3.  Add `GEMINI_API_KEY` and other variables in Railway Service Settings.
4.  Attach a PostgreSQL plugin (optional but recommended for production).
5.  Deploy!

## 📄 License
MIT
