"""
Gunicorn configuration for running the Flask app behind a TLS-terminating reverse proxy.

Environment-driven defaults to work well on PaaS (e.g., Railway, Heroku, Fly.io).
"""

import multiprocessing
import os


def _cpu_workers(default: int = 2) -> int:
    try:
        return int(os.getenv("WEB_CONCURRENCY", str(default)))
    except ValueError:
        return default


bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Worker model and concurrency
workers = _cpu_workers(multiprocessing.cpu_count() * 2 + 1)
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")
threads = int(os.getenv("WEB_THREADS", "1"))

# Timeouts and connection keepalive
timeout = int(os.getenv("WEB_TIMEOUT", "60"))
graceful_timeout = int(os.getenv("WEB_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("WEB_KEEPALIVE", "5"))

# Logging to stdout/stderr (12-factor)
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"

# Trust proxy headers for scheme/host/port/IP
forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "*")
proxy_allow_ips = forwarded_allow_ips

# Ensure https scheme when header is set by the proxy
secure_scheme_headers = {
    "X-Forwarded-Proto": "https",
}


