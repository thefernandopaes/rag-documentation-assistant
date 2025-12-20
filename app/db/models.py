"""
Async SQLAlchemy ORM Models

Async versions of database models maintaining complete schema compatibility
with existing sync models. Works with existing Alembic migrations.

Models:
- Conversation: Chat conversation history
- DocumentChunk: Documentation chunks for RAG
- RateLimit: Rate limiting tracker
"""

from datetime import datetime
import uuid
from sqlalchemy import String, Text, Float, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class Conversation(Base):
    """
    Conversation model - stores chat interactions.

    Compatible with existing 'conversation' table.
    """
    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    ai_response: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, session_id={self.session_id})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "ai_response": self.ai_response,
            "sources": self.sources,
            "response_time": self.response_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "feedback": self.feedback,
        }


class DocumentChunk(Base):
    """
    DocumentChunk model - stores processed documentation chunks.

    Compatible with existing 'document_chunk' table.
    """
    __tablename__ = "document_chunk"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    doc_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    embedding_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, source_url={self.source_url[:50]})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "source_url": self.source_url,
            "title": self.title,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "chunk_index": self.chunk_index,
            "doc_type": self.doc_type,
            "version": self.version,
            "embedding_id": self.embedding_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class RateLimit(Base):
    """
    RateLimit model - tracks API rate limiting per IP.

    Compatible with existing 'rate_limit' table.
    """
    __tablename__ = "rate_limit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False, unique=True)
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_request: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    reset_time: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    def __repr__(self) -> str:
        return f"<RateLimit(ip={self.ip_address}, count={self.request_count})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "ip_address": self.ip_address,
            "request_count": self.request_count,
            "last_request": self.last_request.isoformat() if self.last_request else None,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
        }


# Export all models
__all__ = ["Conversation", "DocumentChunk", "RateLimit", "Base"]
