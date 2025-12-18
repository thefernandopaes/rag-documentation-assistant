"""
Development Server Runner

Runs uvicorn with hot reload for local development.

Usage:
    python run_dev.py
"""

import os
import sys
import uvicorn

# Set development environment
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

if __name__ == "__main__":
    print("=" * 60)
    print("Starting FastAPI Development Server")
    print("=" * 60)
    print("Environment: development")
    print("Hot reload: ENABLED")
    print("URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("\nPress CTRL+C to stop\n")

    # Run uvicorn with development settings
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # Enable hot reload
        reload_dirs=["."],     # Watch current directory
        log_level="info",
        access_log=True,
        workers=1,             # Single worker for development
        proxy_headers=False,   # Not behind proxy in dev
        timeout_keep_alive=5
    )
