"""
FastAPI Dependencies

Provides dependency injection for:
- Rate limiting (async)
- Security validation
- Admin authentication
- Resource injection (RAG engine, database)
- Session management
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_async_db
from app.db.models import RateLimit
from config import Config
from app.utils.validators import validate_query, ValidationError

logger = logging.getLogger(__name__)

# API Key header for admin endpoints
api_key_header = APIKeyHeader(name="X-Admin-Api-Key", auto_error=False)


async def get_rag_engine(request: Request):
    """
    Get RAG engine from application state.

    FastAPI dependency for injecting RAG engine into routes.
    """
    if not hasattr(request.app.state, 'rag_engine'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    return request.app.state.rag_engine


async def get_session_id(request: Request) -> str:
    """
    Get or create session ID from cookies/headers.

    FastAPI dependency for session management.
    """
    import uuid

    # Try to get from cookie
    session_id = request.cookies.get('session_id')

    # Try to get from header
    if not session_id:
        session_id = request.headers.get('X-Session-ID')

    # Generate new if not found
    if not session_id:
        session_id = str(uuid.uuid4())

    return session_id


async def validate_rate_limit(
    request: Request,
    db: AsyncSession = Depends(get_async_db)
) -> None:
    """
    Validate rate limiting for API requests.

    Tracks requests per IP address with configurable limits.
    Raises HTTPException if rate limit exceeded.
    """
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # For localhost/development, skip rate limiting
        if client_ip in ["127.0.0.1", "localhost", "::1"]:
            if not Config._is_production():
                return

        current_time = datetime.utcnow()

        # Get or create rate limit record
        result = await db.execute(
            select(RateLimit).where(RateLimit.ip_address == client_ip)
        )
        rate_record = result.scalar_one_or_none()

        if rate_record:
            # Check if reset time has passed
            if current_time >= rate_record.reset_time:
                # Reset counter
                rate_record.request_count = 1
                rate_record.last_request = current_time
                rate_record.reset_time = current_time + timedelta(minutes=1)
                await db.commit()
                logger.debug(f"Rate limit reset for IP: {client_ip}")
                return

            # Check if limit exceeded
            if rate_record.request_count >= Config.RATE_LIMIT_PER_MINUTE:
                # Check burst allowance
                seconds_since_last = (current_time - rate_record.last_request).total_seconds()

                if seconds_since_last < 1:  # Less than 1 second
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "Rate limit exceeded",
                            "limit": Config.RATE_LIMIT_PER_MINUTE,
                            "reset_at": rate_record.reset_time.isoformat(),
                            "retry_after": int((rate_record.reset_time - current_time).total_seconds())
                        }
                    )

            # Increment counter
            rate_record.request_count += 1
            rate_record.last_request = current_time
            await db.commit()

        else:
            # Create new rate limit record
            rate_record = RateLimit(
                ip_address=client_ip,
                request_count=1,
                last_request=current_time,
                reset_time=current_time + timedelta(minutes=1)
            )
            db.add(rate_record)
            await db.commit()
            logger.debug(f"Created new rate limit record for IP: {client_ip}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rate limiting: {e}")
        # Don't block requests on rate limit errors in non-production
        if Config._is_production():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Rate limiting service unavailable"
            )


async def validate_query_security(query: str) -> str:
    """
    Validate query for security issues.

    Uses validators from utils/validators.py to check for:
    - XSS patterns
    - SQL injection attempts
    - Excessive special characters
    - Character spam

    Returns cleaned query or raises HTTPException.
    """
    try:
        validation_result = validate_query(query)

        if not validation_result['valid']:
            errors = validation_result.get('errors', [])
            error_message = '; '.join(errors)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid query",
                    "message": error_message,
                    "validation_errors": errors
                }
            )

        # Return cleaned query
        return validation_result['query']

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query validation failed"
        )


async def validate_admin_key(
    api_key: Optional[str] = Depends(api_key_header)
) -> None:
    """
    Validate admin API key for protected endpoints.

    In production, requires X-Admin-Api-Key header.
    In development, allows access without key.
    """
    # Skip in development
    if not Config._is_production():
        return

    # Require API key in production
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if api_key != Config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key"
        )


async def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check for proxy headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback to direct client
    return request.client.host if request.client else "unknown"


async def validate_conversation_exists(
    conversation_id: str,
    db: AsyncSession = Depends(get_async_db)
) -> bool:
    """
    Validate that a conversation exists in the database.

    Returns True if exists, raises HTTPException if not found.
    """
    from app.db.models import Conversation

    try:
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )

        return True

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate conversation"
        )


# Dependency combinations for common use cases

async def chat_dependencies(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    _rate_limit: None = Depends(validate_rate_limit),
    session_id: str = Depends(get_session_id),
    rag_engine = Depends(get_rag_engine)
):
    """
    Combined dependencies for chat endpoint.

    Returns: (db, session_id, rag_engine)
    """
    return db, session_id, rag_engine


async def admin_dependencies(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    _admin_key: None = Depends(validate_admin_key),
    rag_engine = Depends(get_rag_engine)
):
    """
    Combined dependencies for admin endpoints.

    Returns: (db, rag_engine)
    """
    return db, rag_engine


# Health check dependency (no auth required)
async def health_check_db(db: AsyncSession = Depends(get_async_db)) -> bool:
    """
    Check database health.

    Returns True if database is accessible.
    """
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
