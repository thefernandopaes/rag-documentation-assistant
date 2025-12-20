"""
Uvicorn Server Configuration for Production

Optimized settings for FastAPI async application:
- Worker processes based on CPU count
- Connection limits and timeouts
- Proxy header handling for reverse proxies
- Graceful shutdown handling
"""

import os
import multiprocessing
from typing import Dict, Any


def get_workers() -> int:
    """
    Calculate optimal number of worker processes.

    Formula: (2 x CPU cores) + 1

    Can be overridden with WEB_CONCURRENCY environment variable.
    """
    try:
        workers_env = os.getenv("WEB_CONCURRENCY")
        if workers_env:
            return int(workers_env)
    except (ValueError, TypeError):
        pass

    # Default: 2 * CPU + 1 (Gunicorn formula)
    cpu_count = multiprocessing.cpu_count()
    return (2 * cpu_count) + 1


def get_host() -> str:
    """Get host address (default: 0.0.0.0 for all interfaces)"""
    return os.getenv("UVICORN_HOST", "0.0.0.0")


def get_port() -> int:
    """Get port number (default: 8000, or from PORT env var)"""
    try:
        return int(os.getenv("PORT", "8000"))
    except (ValueError, TypeError):
        return 8000


def get_log_level() -> str:
    """Get log level (default: info)"""
    return os.getenv("UVICORN_LOG_LEVEL", "info")


def get_config() -> Dict[str, Any]:
    """
    Get complete uvicorn configuration.

    Returns:
        Configuration dictionary for uvicorn.run()
    """
    return {
        # Server binding
        "host": get_host(),
        "port": get_port(),

        # Workers
        "workers": get_workers(),

        # Timeouts
        "timeout_keep_alive": 5,  # Keep-alive timeout (seconds)
        "timeout_notify": 30,     # Worker timeout notification (seconds)
        "timeout_graceful_shutdown": 15,  # Graceful shutdown timeout

        # Logging
        "log_level": get_log_level(),
        "access_log": True,
        "log_config": None,  # Use default logging config

        # Proxy headers (trust X-Forwarded-* headers from reverse proxy)
        "proxy_headers": True,
        "forwarded_allow_ips": "*",  # Trust all proxies (be specific in production!)

        # Connection limits
        "limit_concurrency": 1000,  # Max concurrent connections
        "limit_max_requests": 10000,  # Max requests before worker restart (prevent memory leaks)
        "backlog": 2048,  # Socket backlog size

        # Performance
        "lifespan": "on",  # Enable lifespan events (startup/shutdown)
        "reload": False,   # Disable in production (use for development only)

        # SSL (if certificates provided)
        "ssl_keyfile": os.getenv("SSL_KEYFILE"),
        "ssl_certfile": os.getenv("SSL_CERTFILE"),

        # HTTP
        "server_header": False,  # Don't send Server header (security)
        "date_header": True,     # Send Date header (HTTP/1.1 spec)
    }


def print_config():
    """Print current configuration (for debugging)"""
    config = get_config()

    print("=" * 60)
    print("UVICORN CONFIGURATION")
    print("=" * 60)
    print(f"Host: {config['host']}")
    print(f"Port: {config['port']}")
    print(f"Workers: {config['workers']}")
    print(f"Log level: {config['log_level']}")
    print(f"Proxy headers: {config['proxy_headers']}")
    print(f"Keep-alive timeout: {config['timeout_keep_alive']}s")
    print(f"Max concurrent connections: {config['limit_concurrency']}")
    print(f"Max requests per worker: {config['limit_max_requests']}")
    print("=" * 60)


if __name__ == "__main__":
    # Print configuration when run directly
    print_config()

    # Example usage:
    # import uvicorn
    # from fastapi_app import app
    # config = get_config()
    # uvicorn.run(app, **config)
