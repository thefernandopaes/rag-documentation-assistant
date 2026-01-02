# Flask/Gunicorn (old):
# web: gunicorn -c gunicorn.conf.py app:app

# FastAPI/Uvicorn (new):
web: uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers --forwarded-allow-ips "*"

