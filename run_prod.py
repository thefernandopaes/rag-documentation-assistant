"""
Production Server Runner

Runs uvicorn with production-optimized settings.

Usage:
    python run_prod.py

Environment Variables:
    PORT               - Server port (default: 8000)
    WEB_CONCURRENCY    - Number of workers (default: 2*CPU+1)
    UVICORN_HOST       - Host address (default: 0.0.0.0)
    UVICORN_LOG_LEVEL  - Log level (default: info)
"""

import os
import sys
import uvicorn
from uvicorn_config import get_config, print_config

# Set production environment
os.environ.setdefault("ENVIRONMENT", "production")

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("Starting FastAPI Production Server")
    print("=" * 60)

    # Get and print configuration
    config = get_config()
    print_config()

    print("\nPress CTRL+C to stop\n")

    # Import app
    from fastapi_app import app

    # Run with production configuration
    uvicorn.run(app, **config)
