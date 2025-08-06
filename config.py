import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = "docrag_embeddings"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Cache Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
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
        return True
