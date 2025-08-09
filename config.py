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
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    DOC_USE_SAMPLE = (os.getenv("DOC_USE_SAMPLE", "true").lower() == "true")
    DOC_MAX_PAGES_PER_SOURCE = int(os.getenv("DOC_MAX_PAGES_PER_SOURCE", "50"))
    DOC_CRAWL_TIMEOUT = int(os.getenv("DOC_CRAWL_TIMEOUT", "15"))
    DOC_CRAWL_DELAY = float(os.getenv("DOC_CRAWL_DELAY", "0.5"))
    
    # Cache Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///docrag.db")
    DB_SSLMODE = os.getenv("DB_SSLMODE")  # e.g., require, disable, verify-full
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    
    # Documentation Sources
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
        }
    }
    
    # Response Configuration
    MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "2000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
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
