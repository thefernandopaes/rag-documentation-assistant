# Flask/Gunicorn (old):
# web: gunicorn -c gunicorn.conf.py app:app

# FastAPI/Uvicorn (new):
web: uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --workers ${WEB_CONCURRENCY:-4} --timeout-keep-alive 5 --proxy-headers --forwarded-allow-ips "*" --limit-concurrency 1000

