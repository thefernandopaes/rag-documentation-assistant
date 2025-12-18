import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
    # Session secret (must be set in production)
    SESSION_SECRET = os.getenv("SESSION_SECRET")
    # Admin API key to protect administrative endpoints in production
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = "docrag_embeddings"
    
    # Rate Limiting (Enhanced for API documentation workloads)
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "15"))
    RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "5"))
    
    # Document Processing (Optimized for API documentation)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))  # Larger chunks for API documentation
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    DOC_USE_SAMPLE = (os.getenv("DOC_USE_SAMPLE", "true").lower() == "true")
    DOC_MAX_PAGES_PER_SOURCE = int(os.getenv("DOC_MAX_PAGES_PER_SOURCE", "100"))  # More pages for API docs
    DOC_CRAWL_TIMEOUT = int(os.getenv("DOC_CRAWL_TIMEOUT", "30"))  # Longer timeout for API discovery
    DOC_CRAWL_DELAY = float(os.getenv("DOC_CRAWL_DELAY", "1.0"))  # Respectful crawling
    
    # API-Specific Configuration
    API_DISCOVERY_ENABLED = (os.getenv("API_DISCOVERY_ENABLED", "true").lower() == "true")
    API_CACHE_SPEC_TTL = int(os.getenv("API_CACHE_SPEC_TTL", "86400"))  # 24 hours
    API_MAX_ENDPOINTS_PER_SPEC = int(os.getenv("API_MAX_ENDPOINTS_PER_SPEC", "200"))
    CODE_EXAMPLES_LANGUAGES = os.getenv("CODE_EXAMPLES_LANGUAGES", "curl,python,javascript,nodejs,php").split(",")
    
    # Cache Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///docrag.db")
    DB_SSLMODE = os.getenv("DB_SSLMODE")  # e.g., require, disable, verify-full
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    
    # API Documentation Sources (Updated for API-focused system)
    API_DOC_SOURCES = {
        "stripe": {
            "base_url": "https://stripe.com/docs/api",
            "openapi_url": "https://raw.githubusercontent.com/stripe/openapi/master/openapi/spec3.json",
            "type": "rest_api"
        },
        "github": {
            "base_url": "https://docs.github.com/en/rest",
            "openapi_url": "https://github.com/github/rest-api-description/raw/main/descriptions/api.github.com/api.github.com.json",
            "type": "rest_api"
        },
        "openai": {
            "base_url": "https://platform.openai.com/docs/api-reference",
            "openapi_url": "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml",
            "type": "rest_api"
        },
        "twilio": {
            "base_url": "https://www.twilio.com/docs/usage/api",
            "type": "rest_api"
        },
        "discord": {
            "base_url": "https://discord.com/developers/docs/intro",
            "type": "rest_api"
        }
    }
    
    # Documentation sources (expanded for technical documentation platform)
    DOC_SOURCES = {
        "react": {
            "base_url": "https://react.dev/",
            "docs_url": "https://react.dev/learn",
            "type": "react"
        },
        "python": {
            "base_url": "https://docs.python.org/3/",
            "docs_url": "https://docs.python.org/3/tutorial/",
            "type": "python"
        },
        "fastapi": {
            "base_url": "https://fastapi.tiangolo.com/",
            "docs_url": "https://fastapi.tiangolo.com/tutorial/",
            "type": "fastapi"
        },
        "docker": {
            "base_url": "https://docs.docker.com/",
            "docs_url": "https://docs.docker.com/get-started/",
            "type": "docker"
        },
        "aws_lambda": {
            "base_url": "https://docs.aws.amazon.com/lambda/",
            "docs_url": "https://docs.aws.amazon.com/lambda/latest/dg/getting-started.html",
            "type": "aws_lambda"
        }
    }
    
    # Response Configuration (Enhanced for API documentation)
    MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "3000"))  # More tokens for detailed API responses
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))  # Lower temperature for more consistent API documentation
    
    # Monitoring and Analytics
    MONITORING_ENABLED = (os.getenv("MONITORING_ENABLED", "false").lower() == "true")
    ANALYTICS_RETENTION_DAYS = int(os.getenv("ANALYTICS_RETENTION_DAYS", "90"))
    ERROR_REPORTING_ENABLED = (os.getenv("ERROR_REPORTING_ENABLED", "true").lower() == "true")
    
    # Security Configuration
    API_KEY_ROTATION_DAYS = int(os.getenv("API_KEY_ROTATION_DAYS", "90"))
    REQUEST_SIZE_LIMIT = os.getenv("REQUEST_SIZE_LIMIT", "16KB")
    ALLOWED_DOMAINS = os.getenv("ALLOWED_DOMAINS", "").split(",") if os.getenv("ALLOWED_DOMAINS") else []
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Enforce SESSION_SECRET in production environments
        if cls._is_production() and not cls.SESSION_SECRET:
            raise ValueError("SESSION_SECRET is required in production environments")

        # Ensure CHROMA_DB_PATH exists and is writable (create if missing)
        cls._ensure_chroma_dir_writable()

        return True

    @classmethod
    def get_database_uri(cls) -> str:
        """Return a SQLAlchemy-compatible database URI, normalizing postgres scheme."""
        uri = cls.DATABASE_URL.strip()
        # Normalize legacy postgres:// to postgresql+psycopg2:// for SQLAlchemy
        if uri.startswith("postgres://"):
            uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)
        return uri

    @classmethod
    def get_sqlalchemy_engine_options(cls) -> dict:
        """Return engine options tailored to the selected database backend."""
        uri = cls.get_database_uri()
        is_postgres = uri.startswith("postgresql://") or uri.startswith("postgresql+psycopg2://")

        # Base options
        engine_options: dict = {
            "pool_recycle": 300,
            "pool_pre_ping": True,
        }

        if is_postgres:
            engine_options.update({
                "pool_size": cls.DB_POOL_SIZE,
                "max_overflow": cls.DB_MAX_OVERFLOW,
            })
            # Optional SSL
            if cls.DB_SSLMODE:
                engine_options["connect_args"] = {"sslmode": cls.DB_SSLMODE}

        return engine_options

    @classmethod
    def _is_production(cls) -> bool:
        """Detect if running in a production environment based on common env vars."""
        env_candidates = {
            (os.getenv("ENV") or "").lower(),
            (os.getenv("FLASK_ENV") or "").lower(),
            (os.getenv("PYTHON_ENV") or "").lower(),
        }
        return "production" in env_candidates

    @classmethod
    def _ensure_chroma_dir_writable(cls) -> None:
        """Ensure the Chroma DB directory exists and is writable; create if necessary."""
        import tempfile

        path = os.path.abspath(cls.CHROMA_DB_PATH)
        os.makedirs(path, exist_ok=True)

        # Verify writability by creating and removing a temp file
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=".chroma_write_test_", dir=path)
            os.close(fd)
            os.remove(tmp_path)
        except Exception as exc:
            raise ValueError(
                f"CHROMA_DB_PATH '{path}' is not writable or cannot be created: {exc}"
            ) from exc
