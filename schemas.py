"""
Pydantic Schemas for FastAPI Request/Response Validation

Includes custom validators for security (XSS, SQL injection, special chars)
migrated from utils/validators.py
"""

import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict


# ============================================================================
# Request Schemas
# ============================================================================

class ChatRequest(BaseModel):
    """
    Chat query request with comprehensive validation.

    Validates:
    - Length constraints (3-500 chars)
    - XSS patterns (script tags, javascript:, event handlers)
    - SQL injection attempts
    - Excessive special characters
    - Repeated character spam
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User query for the RAG system",
        examples=["How do I authenticate with the Stripe API?"]
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation context (UUID format)",
        pattern=r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    @field_validator('query')
    @classmethod
    def validate_query_security(cls, v: str) -> str:
        """
        Comprehensive query validation for security threats.

        Checks for:
        - XSS attacks (script tags, event handlers, JS protocol)
        - SQL injection patterns
        - Excessive special characters
        - Character spam
        """
        # Check for XSS patterns
        xss_patterns = [
            r'<script[^>]*>',     # Script tags
            r'javascript:',        # JavaScript protocol
            r'on\w+\s*=',         # Event handlers (onclick, onload, etc)
            r'eval\s*\(',         # eval() calls
            r'document\.',        # Document object access
            r'window\.',          # Window object access
        ]

        for pattern in xss_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains potentially malicious content (XSS pattern detected)')

        # Check for SQL injection patterns (strict mode)
        sql_patterns = [
            r'\b(union|select|insert|update|delete|drop|alter|create)\b\s+',
            r'--\s*',              # SQL comments
            r'/\*.*\*/',           # SQL block comments
            r';\s*$',              # Semicolon at end
        ]

        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains SQL-like syntax that is not allowed')

        # Check for excessive special characters (>30% of content)
        special_char_count = len(re.findall(r'[^a-zA-Z0-9\s]', v))
        special_char_ratio = special_char_count / len(v) if len(v) > 0 else 0

        if special_char_ratio > 0.3:
            raise ValueError('Query contains too many special characters (>30%)')

        # Check for repeated characters (potential spam)
        if re.search(r'(.)\1{10,}', v):
            raise ValueError('Query contains excessive repeated characters')

        return v


class FeedbackRequest(BaseModel):
    """
    User feedback on AI response.

    Feedback values:
    - 1: Thumbs up (positive)
    - -1: Thumbs down (negative)
    """
    conversation_id: str = Field(
        ...,
        description="Conversation ID (UUID)",
        pattern=r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    )
    feedback: int = Field(
        ...,
        description="Feedback value: 1 (positive) or -1 (negative)",
        ge=-1,
        le=1
    )

    @field_validator('feedback')
    @classmethod
    def validate_feedback_value(cls, v: int) -> int:
        """Ensure feedback is exactly 1 or -1."""
        if v not in [1, -1]:
            raise ValueError('Feedback must be 1 (thumbs up) or -1 (thumbs down)')
        return v


class InitializeRequest(BaseModel):
    """
    System initialization request (admin only).

    Optional fields for controlling initialization behavior.
    """
    force: bool = Field(
        False,
        description="Force re-initialization even if documents exist"
    )
    sources: Optional[List[str]] = Field(
        None,
        description="Specific documentation sources to process (if None, process all)"
    )


# ============================================================================
# Response Schemas
# ============================================================================

class Source(BaseModel):
    """Source reference for RAG response."""
    url: str = Field(..., description="Source URL")
    title: Optional[str] = Field(None, description="Source title")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")


class CodeExample(BaseModel):
    """Code example with language specification."""
    language: str = Field(..., description="Programming language (python, javascript, curl, etc)")
    code: str = Field(..., description="Code snippet")
    description: Optional[str] = Field(None, description="Description of what the code does")


class APIEndpoint(BaseModel):
    """API endpoint information."""
    method: str = Field(..., description="HTTP method (GET, POST, etc)")
    path: str = Field(..., description="Endpoint path")
    description: Optional[str] = Field(None, description="Endpoint description")


class Parameter(BaseModel):
    """API parameter specification."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, integer, etc)")
    required: bool = Field(..., description="Whether parameter is required")
    description: Optional[str] = Field(None, description="Parameter description")


class ErrorCode(BaseModel):
    """API error code information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    description: Optional[str] = Field(None, description="Detailed description")


class ChatResponse(BaseModel):
    """
    Chat response with RAG-generated answer and metadata.

    Includes optional API-specific fields:
    - Code examples in multiple languages
    - API endpoint details
    - Authentication information
    - Parameter specifications
    - Response format
    - Error codes
    - Related concepts
    """
    response: str = Field(..., description="AI-generated response")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source references used for response"
    )
    response_time: float = Field(..., description="Response generation time in seconds")
    cached: bool = Field(False, description="Whether response was cached")

    # API-specific optional fields
    examples: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Code examples in various languages"
    )
    endpoints: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Related API endpoints"
    )
    authentication: Optional[Dict[str, Any]] = Field(
        None,
        description="Authentication requirements"
    )
    parameters: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="API parameters"
    )
    response_format: Optional[Dict[str, Any]] = Field(
        None,
        description="Expected response format"
    )
    error_codes: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Possible error codes"
    )
    related_concepts: Optional[List[str]] = Field(
        None,
        description="Related concepts or questions"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "To authenticate with the Stripe API, use your API key...",
                "sources": [
                    {"url": "https://stripe.com/docs/api/authentication", "title": "Authentication"}
                ],
                "response_time": 1.23,
                "cached": False,
                "examples": [
                    {
                        "language": "python",
                        "code": "import stripe\nstripe.api_key = 'sk_test_...'",
                        "description": "Set API key in Python"
                    }
                ]
            }
        }
    )


class HistoryItem(BaseModel):
    """Single conversation history item."""
    id: str = Field(..., description="Conversation ID")
    query: str = Field(..., description="User query")
    response: str = Field(..., description="AI response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources")
    created_at: str = Field(..., description="Timestamp (ISO format)")
    response_time: float = Field(..., description="Response time in seconds")
    feedback: Optional[int] = Field(None, description="User feedback (1 or -1)")


class HistoryResponse(BaseModel):
    """Conversation history response."""
    history: List[HistoryItem] = Field(default_factory=list, description="Conversation history")
    total: int = Field(..., description="Total conversations in session")


class CollectionStats(BaseModel):
    """ChromaDB collection statistics."""
    document_count: int = Field(..., description="Total documents in collection")
    collection_name: str = Field(..., description="Collection name")


class ConversationStats(BaseModel):
    """Conversation statistics."""
    total: int = Field(..., description="Total conversations")
    avg_response_time: float = Field(..., description="Average response time in seconds")
    positive_feedback: int = Field(..., description="Count of positive feedback")
    negative_feedback: int = Field(..., description="Count of negative feedback")


class CacheStats(BaseModel):
    """Cache statistics."""
    total_entries: int = Field(..., description="Total cache entries")
    hit_rate: Optional[float] = Field(None, description="Cache hit rate (0-1)")
    size_mb: Optional[float] = Field(None, description="Cache size in MB")


class SystemInfo(BaseModel):
    """System information."""
    is_production: bool = Field(..., description="Whether running in production")
    version: str = Field(default="2.0.0", description="API version")


class StatsResponse(BaseModel):
    """System statistics response."""
    documents: Dict[str, Any] = Field(..., description="Document statistics")
    conversations: Dict[str, Any] = Field(..., description="Conversation statistics")
    cache: Dict[str, Any] = Field(..., description="Cache statistics")
    system: Dict[str, Any] = Field(..., description="System information")


class InitializeResponse(BaseModel):
    """System initialization response."""
    message: str = Field(..., description="Status message")
    status: str = Field(..., description="Initialization status")
    document_count: Optional[int] = Field(None, description="Number of documents processed")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    path: Optional[str] = Field(None, description="Request path that caused error")
    status_code: Optional[int] = Field(None, description="HTTP status code")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    framework: str = Field(..., description="Framework name")
    async_enabled: bool = Field(..., alias="async", description="Whether async is enabled")


# ============================================================================
# Utility Validators
# ============================================================================

def validate_uuid(value: str) -> bool:
    """Validate UUID v4 format."""
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$'
    return bool(re.match(uuid_pattern, value, re.IGNORECASE))


def validate_url(url: str) -> bool:
    """Validate URL format (HTTP/HTTPS only)."""
    url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$'
    return bool(re.match(url_pattern, url))


def validate_email(email: str) -> bool:
    """Validate email format."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))
