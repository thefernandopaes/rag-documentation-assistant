from app import db
from datetime import datetime
import uuid

class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), nullable=False)
    user_query = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text)  # JSON string of sources
    response_time = db.Column(db.Float)  # Response time in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.Integer)  # 1 for thumbs up, -1 for thumbs down

class DocumentChunk(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_url = db.Column(db.String(500), nullable=False)
    title = db.Column(db.String(200))
    content = db.Column(db.Text, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    doc_type = db.Column(db.String(50))  # 'react', 'python', 'fastapi'
    version = db.Column(db.String(20))
    embedding_id = db.Column(db.String(100))  # ChromaDB document ID
    content_hash = db.Column(db.String(64))   # Idempotency key (SHA-256)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RateLimit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(45), nullable=False, unique=True)
    request_count = db.Column(db.Integer, default=0)
    last_request = db.Column(db.DateTime, default=datetime.utcnow)
    reset_time = db.Column(db.DateTime, default=datetime.utcnow)
