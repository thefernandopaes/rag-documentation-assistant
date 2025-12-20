"""
Memory Schemas - Pydantic Models for Memory Endpoints

Request/response validation for conversation memory API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional


# ============================================================================
# MEMORY CONFIGURATION
# ============================================================================

class MemoryConfigRequest(BaseModel):
    """Request to update memory configuration"""

    strategy: Optional[str] = Field(
        None,
        description="Memory strategy: buffer_window | summary | token_buffer"
    )

    buffer_window_size: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Number of messages to keep (for buffer_window strategy)"
    )

    token_limit: Optional[int] = Field(
        None,
        ge=100,
        le=10000,
        description="Maximum tokens to keep (for token_buffer strategy)"
    )

    summary_interval: Optional[int] = Field(
        None,
        ge=5,
        le=50,
        description="Messages before summarizing (for summary strategy)"
    )

    @validator('strategy')
    def validate_strategy(cls, v):
        """Validate strategy is one of allowed values"""
        if v is not None:
            allowed = ['buffer_window', 'summary', 'token_buffer']
            if v not in allowed:
                raise ValueError(f"Strategy must be one of: {', '.join(allowed)}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "strategy": "buffer_window",
                "buffer_window_size": 20,
                "token_limit": 2000,
                "summary_interval": 10
            }
        }


class MemoryConfigResponse(BaseModel):
    """Memory configuration response"""

    session_id: str = Field(..., description="Session identifier")

    strategy: str = Field(
        ...,
        description="Current memory strategy"
    )

    buffer_window_size: int = Field(
        ...,
        description="Buffer window size"
    )

    token_limit: int = Field(
        ...,
        description="Token limit for token_buffer strategy"
    )

    summary_interval: int = Field(
        ...,
        description="Summary interval for summary strategy"
    )

    class Config:
        schema_extra = {
            "example": {
                "session_id": "abc123-def456",
                "strategy": "buffer_window",
                "buffer_window_size": 20,
                "token_limit": 2000,
                "summary_interval": 10
            }
        }


# ============================================================================
# MESSAGE HISTORY
# ============================================================================

class Message(BaseModel):
    """Single conversation message"""

    role: str = Field(
        ...,
        description="Message role: user | assistant | system"
    )

    content: str = Field(
        ...,
        description="Message content"
    )

    class Config:
        schema_extra = {
            "example": {
                "role": "user",
                "content": "What is FastAPI?"
            }
        }


class MessageHistoryResponse(BaseModel):
    """Memory messages response"""

    session_id: str = Field(..., description="Session identifier")

    messages: List[Dict] = Field(
        ...,
        description="List of conversation messages"
    )

    count: int = Field(
        ...,
        description="Total number of messages returned"
    )

    class Config:
        schema_extra = {
            "example": {
                "session_id": "abc123-def456",
                "messages": [
                    {"role": "user", "content": "What is FastAPI?"},
                    {"role": "assistant", "content": "FastAPI is a modern web framework..."}
                ],
                "count": 2
            }
        }


# ============================================================================
# MEMORY STATISTICS
# ============================================================================

class MemoryStatsResponse(BaseModel):
    """Memory statistics response"""

    total_messages: int = Field(
        ...,
        description="Total number of messages in memory"
    )

    total_tokens: int = Field(
        ...,
        description="Total tokens used by messages"
    )

    strategy: str = Field(
        ...,
        description="Current memory strategy"
    )

    config: Dict = Field(
        ...,
        description="Memory configuration details"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_messages": 42,
                "total_tokens": 1250,
                "strategy": "buffer_window",
                "config": {
                    "buffer_window_size": 20,
                    "token_limit": 2000,
                    "summary_interval": 10
                }
            }
        }


# ============================================================================
# OPERATIONS
# ============================================================================

class ClearMemoryResponse(BaseModel):
    """Response for clear memory operation"""

    message: str = Field(
        ...,
        description="Success message"
    )

    session_id: str = Field(
        ...,
        description="Session identifier"
    )

    class Config:
        schema_extra = {
            "example": {
                "message": "Memory cleared successfully",
                "session_id": "abc123-def456"
            }
        }
