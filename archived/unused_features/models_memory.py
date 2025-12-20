"""
Memory Models - SQLAlchemy Models for Conversation Memory

Database models for storing conversation history and memory configuration.
"""

from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

# Import Base from models_async to use same declarative base
try:
    from models_async import Base
except ImportError:
    # Fallback if models_async not available yet
    Base = declarative_base()


class ConversationMemory(Base):
    """
    Stores conversation messages for memory management.

    Supports different memory strategies:
    - buffer_window: Keep last N messages
    - summary: Summarize old messages
    - token_buffer: Keep messages within token limit
    """
    __tablename__ = "conversation_memory"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Session tracking
    session_id = Column(String(36), nullable=False, index=True)

    # Message content
    role = Column(String(20), nullable=False)  # 'user', 'assistant', or 'system'
    content = Column(Text, nullable=False)
    tokens = Column(Integer, default=0)  # Estimated token count

    # Metadata
    message_index = Column(Integer, nullable=False)  # Order in conversation
    memory_strategy = Column(String(50), default='buffer_window')
    is_summarized = Column(Boolean, default=False)
    summary_of = Column(String(500), nullable=True)  # IDs of messages this summarizes

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Composite indexes for fast retrieval
    __table_args__ = (
        Index('idx_session_strategy', 'session_id', 'memory_strategy'),
        Index('idx_session_index', 'session_id', 'message_index'),
    )

    def __repr__(self):
        return f"<ConversationMemory(id={self.id}, session={self.session_id}, role={self.role}, index={self.message_index})>"


class MemoryConfiguration(Base):
    """
    User/session-specific memory configuration.

    Allows customization of memory strategies per session.
    """
    __tablename__ = "memory_configuration"

    # Primary key is session_id (one config per session)
    session_id = Column(String(36), primary_key=True)

    # Strategy selection
    strategy = Column(
        String(50),
        default='buffer_window',
        nullable=False
    )  # 'buffer_window', 'summary', or 'token_buffer'

    # Strategy parameters
    buffer_window_size = Column(Integer, default=20)  # For buffer_window: last N messages
    token_limit = Column(Integer, default=2000)  # For token_buffer: max tokens to keep
    summary_interval = Column(Integer, default=10)  # For summary: messages before summarizing

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MemoryConfiguration(session={self.session_id}, strategy={self.strategy})>"
