# Configuration Management

This document outlines how configuration is managed in the DocRag application, including environment variables, configuration patterns, and secrets management.

## Configuration Architecture

The DocRag application uses a centralized configuration system based on environment variables with sensible defaults. All configuration is managed through the `Config` class in `config.py`.

### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **`.env` file** (loaded automatically)
3. **Default Values** (defined in Config class)

## Environment Variables

### Core Secrets

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `OPENAI_API_KEY` | string | Yes | OpenAI API key for embeddings and completions |
| `SESSION_SECRET` | string | Prod only | Flask session secret key (64 hex chars) |
| `ADMIN_API_KEY` | string | Prod only | API key for administrative endpoints |

### Database Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | string | `sqlite:///docrag.db` | Database connection string |
| `DB_SSLMODE` | string | None | PostgreSQL SSL mode (require, disable, verify-full) |
| `DB_POOL_SIZE` | int | `5` | Connection pool size for PostgreSQL |
| `DB_MAX_OVERFLOW` | int | `10` | Maximum overflow connections |

### RAG Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CHROMA_DB_PATH` | string | `./chroma_db` | ChromaDB storage path |
| `CHUNK_SIZE` | int | `1000` | Text chunk size for processing |
| `CHUNK_OVERLAP` | int | `200` | Chunk overlap size |
| `MAX_RESPONSE_TOKENS` | int | `2000` | Maximum tokens in responses |
| `TEMPERATURE` | float | `0.7` | OpenAI model temperature |

### Document Processing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOC_USE_SAMPLE` | bool | `true` | Use sample data instead of crawling |
| `DOC_MAX_PAGES_PER_SOURCE` | int | `50` | Maximum pages to crawl per source |
| `DOC_CRAWL_DELAY` | float | `0.5` | Delay between requests (seconds) |
| `DOC_CRAWL_TIMEOUT` | int | `15` | Request timeout (seconds) |

### Performance & Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RATE_LIMIT_PER_MINUTE` | int | `10` | API requests per minute per IP |
| `CACHE_TTL` | int | `3600` | Cache time-to-live (seconds) |
| `OPENAI_TIMEOUT` | int | `30` | OpenAI API timeout (seconds) |
| `OPENAI_MAX_RETRIES` | int | `2` | Maximum retry attempts |
| `REQUESTS_MAX_RETRIES` | int | `2` | HTTP request retry attempts |
| `REQUESTS_BACKOFF_FACTOR` | float | `0.5` | Exponential backoff factor |

### Deployment Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENV` | string | `development` | Environment (production, development) |
| `WEB_CONCURRENCY` | int | `2` | Gunicorn worker processes |
| `WEB_THREADS` | int | `1` | Gunicorn threads per worker |
| `LOG_LEVEL` | string | `info` | Logging level |
| `FORWARDED_ALLOW_IPS` | string | `*` | Trusted proxy IPs |

## Configuration Patterns

### 1. Environment Detection

```python
@classmethod
def _is_production(cls) -> bool:
    """Detect if running in production environment."""
    env_candidates = {
        (os.getenv("ENV") or "").lower(),
        (os.getenv("FLASK_ENV") or "").lower(),
        (os.getenv("PYTHON_ENV") or "").lower(),
    }
    return "production" in env_candidates
```

### 2. Configuration Validation

```python
@classmethod
def validate_config(cls):
    """Validate required configuration on startup."""
    if not cls.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    if cls._is_production() and not cls.SESSION_SECRET:
        raise ValueError("SESSION_SECRET is required in production")
    
    cls._ensure_chroma_dir_writable()
    return True
```

### 3. Database URI Normalization

```python
@classmethod
def get_database_uri(cls) -> str:
    """Normalize database URI for SQLAlchemy compatibility."""
    uri = cls.DATABASE_URL.strip()
    # Handle legacy postgres:// scheme
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)
    return uri
```

### 4. Engine Options by Database Type

```python
@classmethod
def get_sqlalchemy_engine_options(cls) -> dict:
    """Return database-specific engine options."""
    uri = cls.get_database_uri()
    is_postgres = uri.startswith("postgresql://") or uri.startswith("postgresql+psycopg2://")
    
    engine_options = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    
    if is_postgres:
        engine_options.update({
            "pool_size": cls.DB_POOL_SIZE,
            "max_overflow": cls.DB_MAX_OVERFLOW,
        })
        if cls.DB_SSLMODE:
            engine_options["connect_args"] = {"sslmode": cls.DB_SSLMODE}
    
    return engine_options
```

## Environment File Management

### Development (.env)

Create a `.env` file for local development:

```env
# --- Secrets ---
OPENAI_API_KEY=sk-your-openai-api-key
SESSION_SECRET=your-64-hex-secret
ADMIN_API_KEY=your-admin-key

# --- Database ---
DATABASE_URL=sqlite:///docrag.db

# --- ChromaDB ---
CHROMA_DB_PATH=./chroma_db

# --- Development Settings ---
ENV=development
DOC_USE_SAMPLE=true
LOG_LEVEL=debug
```

### Production Environment Variables

Set directly in your deployment platform:

```env
# --- Secrets (REQUIRED) ---
OPENAI_API_KEY=sk-prod-key
SESSION_SECRET=64-hex-production-secret
ADMIN_API_KEY=secure-admin-key

# --- Database ---
DATABASE_URL=postgresql://user:password@host:5432/dbname
DB_SSLMODE=require
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# --- Storage ---
CHROMA_DB_PATH=/var/lib/docrag/chroma_db

# --- Production Settings ---
ENV=production
DOC_USE_SAMPLE=false
WEB_CONCURRENCY=4
LOG_LEVEL=info
```

### Environment Template (.env.example)

The `.env.example` file serves as documentation and template:

```env
# --- Secrets ---
OPENAI_API_KEY=your-openai-api-key-here
SESSION_SECRET=your-64-hex-secret-here
ADMIN_API_KEY=your-64-hex-admin-key-here

# --- DB ---
DATABASE_URL=postgresql://user:password@host:5432/dbname
DB_SSLMODE=require

# --- ChromaDB ---
CHROMA_DB_PATH=/var/lib/docrag/chroma_db

# --- RAG: chunking ---
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

## Configuration Best Practices

### 1. Secret Management

**Do:**
- Use environment variables for all secrets
- Keep `.env` out of version control
- Use `.env.example` with placeholder values
- Validate required secrets on startup
- Rotate secrets regularly

**Don't:**
- Commit real secrets to version control
- Log sensitive configuration values
- Use default secrets in production
- Store secrets in configuration files

### 2. Environment-Specific Configuration

```python
# Good: Environment-aware configuration
if Config._is_production():
    app.config.update(
        PREFERRED_URL_SCHEME='https',
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
    )

# Bad: Hardcoded environment assumptions
app.config['SESSION_COOKIE_SECURE'] = True  # Breaks local development
```

### 3. Default Values

```python
# Good: Sensible defaults with environment override
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

# Bad: Required environment variable without default
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE"))  # KeyError if missing
```

### 4. Type Conversion

```python
# Good: Explicit type conversion with error handling
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
except ValueError:
    raise ValueError("CHUNK_SIZE must be a valid integer")

# Bad: Implicit string values
CHUNK_SIZE = os.getenv("CHUNK_SIZE", "1000")  # Returns string, not int
```

## Configuration Validation

### Startup Validation

The application validates configuration during startup:

```python
def create_app():
    # Validate configuration before proceeding
    Config.validate_config()
    
    app = Flask(__name__)
    # ... rest of app initialization
```

### Custom Validators

```python
@classmethod
def _ensure_chroma_dir_writable(cls) -> None:
    """Ensure ChromaDB directory exists and is writable."""
    import tempfile
    
    path = os.path.abspath(cls.CHROMA_DB_PATH)
    os.makedirs(path, exist_ok=True)
    
    # Test writability
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".write_test_", dir=path)
        os.close(fd)
        os.remove(tmp_path)
    except Exception as exc:
        raise ValueError(f"CHROMA_DB_PATH '{path}' is not writable: {exc}")
```

## Runtime Configuration Changes

### Hot Reloading

Some configuration can be changed at runtime:

```python
# Example: Updating rate limits
@app.route('/api/admin/config', methods=['POST'])
def update_config():
    if not validate_admin_key(request.headers.get('Authorization')):
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    
    # Update runtime configuration
    if 'rate_limit' in data:
        Config.RATE_LIMIT_PER_MINUTE = int(data['rate_limit'])
    
    return jsonify({'status': 'updated'})
```

### Configuration Monitoring

Log configuration changes:

```python
import logging

logger = logging.getLogger(__name__)

def update_runtime_config(key: str, value: Any):
    old_value = getattr(Config, key, None)
    setattr(Config, key, value)
    logger.info(f"Configuration updated: {key} changed from {old_value} to {value}")
```

## Deployment-Specific Configuration

### Railway Platform

```env
# Railway automatically sets these
PORT=5000
RAILWAY_ENVIRONMENT=production

# Custom configuration
ENV=production
DATABASE_URL=${{Postgres.DATABASE_URL}}
CHROMA_DB_PATH=/var/lib/docrag/chroma_db
```

### Docker Container

```dockerfile
# Set configuration via environment variables
ENV ENV=production
ENV CHROMA_DB_PATH=/var/lib/docrag/chroma_db

# Create volume mount points
VOLUME ["/var/lib/docrag"]

# Configure application
COPY .env.production /app/.env
```

### Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: docrag-config
data:
  ENV: "production"
  CHROMA_DB_PATH: "/var/lib/docrag/chroma_db"
  WEB_CONCURRENCY: "4"

---
apiVersion: v1
kind: Secret
metadata:
  name: docrag-secrets
type: Opaque
data:
  OPENAI_API_KEY: base64-encoded-key
  SESSION_SECRET: base64-encoded-secret
```

## Troubleshooting Configuration

### Common Issues

#### 1. Missing Environment Variables

```bash
# Check if variables are loaded
python -c "from config import Config; print(Config.OPENAI_API_KEY)"
```

#### 2. .env File Not Loading

```python
# Debug .env loading
from dotenv import load_dotenv
import os

print(f"Current directory: {os.getcwd()}")
print(f".env exists: {os.path.exists('.env')}")

load_dotenv(verbose=True)  # Shows what's loaded
```

#### 3. Type Conversion Errors

```python
# Debug configuration types
from config import Config

print(f"CHUNK_SIZE type: {type(Config.CHUNK_SIZE)}")
print(f"CHUNK_SIZE value: {Config.CHUNK_SIZE}")
```

#### 4. Database Connection Issues

```python
# Test database configuration
from config import Config

print(f"Database URI: {Config.get_database_uri()}")
print(f"Engine options: {Config.get_sqlalchemy_engine_options()}")
```

### Configuration Health Check

```python
def health_check_config():
    """Verify configuration is valid and accessible."""
    issues = []
    
    # Check required secrets
    if not Config.OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY not configured")
    
    # Check database connectivity
    try:
        from app import db
        db.engine.execute("SELECT 1")
    except Exception as e:
        issues.append(f"Database connection failed: {e}")
    
    # Check ChromaDB path
    if not os.path.exists(Config.CHROMA_DB_PATH):
        issues.append(f"ChromaDB path does not exist: {Config.CHROMA_DB_PATH}")
    
    return {
        'healthy': len(issues) == 0,
        'issues': issues
    }
```

---

*Keep configuration simple, secure, and well-documented. Always validate configuration on startup and provide clear error messages for misconfigurations.*