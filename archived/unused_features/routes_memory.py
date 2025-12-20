"""
Memory Routes - FastAPI Endpoints for Conversation Memory

Provides endpoints for managing conversation memory and configuration.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import logging

from conversation_memory import ConversationMemoryManager, get_memory_manager
from schemas_memory import (
    MemoryConfigRequest,
    MemoryConfigResponse,
    MessageHistoryResponse,
    MemoryStatsResponse,
    ClearMemoryResponse
)
from database_async import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["Memory"])


# ============================================================================
# DEPENDENCIES
# ============================================================================

async def get_session_id(request: Request) -> str:
    """
    Get session ID from cookie or generate new one.

    Args:
        request: FastAPI request

    Returns:
        Session ID
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        # Generate new session ID
        import uuid
        session_id = str(uuid.uuid4())
    return session_id


async def get_memory_manager_dependency(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db)
) -> ConversationMemoryManager:
    """
    Get memory manager for current session.

    Args:
        session_id: Session identifier
        db: Database session

    Returns:
        ConversationMemoryManager instance
    """
    return ConversationMemoryManager(session_id, db)


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

@router.get("/config", response_model=MemoryConfigResponse)
async def get_memory_config(
    session_id: str = Depends(get_session_id),
    manager: ConversationMemoryManager = Depends(get_memory_manager_dependency)
):
    """
    Get current memory configuration for session.

    Returns the memory strategy and parameters currently in use.

    **Example Response:**
    ```json
    {
        "session_id": "abc123-def456",
        "strategy": "buffer_window",
        "buffer_window_size": 20,
        "token_limit": 2000,
        "summary_interval": 10
    }
    ```

    Args:
        session_id: Session identifier (from cookie)
        manager: Memory manager instance

    Returns:
        Current memory configuration
    """
    try:
        config = await manager.get_memory_config()

        return MemoryConfigResponse(
            session_id=session_id,
            strategy=config.strategy,
            buffer_window_size=config.buffer_window_size,
            token_limit=config.token_limit,
            summary_interval=config.summary_interval
        )

    except Exception as e:
        logger.error(f"Error getting memory config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory configuration: {str(e)}"
        )


@router.put("/config", response_model=MemoryConfigResponse)
async def update_memory_config(
    request: MemoryConfigRequest,
    session_id: str = Depends(get_session_id),
    manager: ConversationMemoryManager = Depends(get_memory_manager_dependency)
):
    """
    Update memory configuration for session.

    Allows customization of memory strategy and parameters.

    **Example Request:**
    ```json
    {
        "strategy": "token_buffer",
        "token_limit": 3000
    }
    ```

    **Strategies:**
    - `buffer_window`: Keep last N messages (simple, fast)
    - `summary`: Summarize old messages (context-preserving)
    - `token_buffer`: Keep messages within token limit (precise)

    Args:
        request: Configuration update request
        session_id: Session identifier (from cookie)
        manager: Memory manager instance

    Returns:
        Updated memory configuration

    Raises:
        HTTPException: If update fails
    """
    try:
        config = await manager.get_memory_config()

        # Update config fields if provided
        if request.strategy is not None:
            config.strategy = request.strategy

        if request.buffer_window_size is not None:
            config.buffer_window_size = request.buffer_window_size

        if request.token_limit is not None:
            config.token_limit = request.token_limit

        if request.summary_interval is not None:
            config.summary_interval = request.summary_interval

        # Commit changes
        await manager.db.commit()
        await manager.db.refresh(config)

        logger.info(f"Updated memory config for session {session_id}: strategy={config.strategy}")

        return MemoryConfigResponse(
            session_id=session_id,
            strategy=config.strategy,
            buffer_window_size=config.buffer_window_size,
            token_limit=config.token_limit,
            summary_interval=config.summary_interval
        )

    except Exception as e:
        logger.error(f"Error updating memory config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update memory configuration: {str(e)}"
        )


# ============================================================================
# MESSAGE RETRIEVAL ENDPOINTS
# ============================================================================

@router.get("/messages", response_model=MessageHistoryResponse)
async def get_memory_messages(
    limit: int = 20,
    session_id: str = Depends(get_session_id),
    manager: ConversationMemoryManager = Depends(get_memory_manager_dependency)
):
    """
    Get conversation messages from memory.

    Retrieves messages according to the current memory strategy.

    **Query Parameters:**
    - `limit`: Maximum number of messages to return (default: 20)

    **Example Response:**
    ```json
    {
        "session_id": "abc123-def456",
        "messages": [
            {"role": "user", "content": "What is FastAPI?"},
            {"role": "assistant", "content": "FastAPI is a modern web framework..."}
        ],
        "count": 2
    }
    ```

    Args:
        limit: Maximum messages to return
        session_id: Session identifier (from cookie)
        manager: Memory manager instance

    Returns:
        Message history response

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        messages = await manager.get_messages(limit=limit)

        logger.info(f"Retrieved {len(messages)} messages for session {session_id}")

        return MessageHistoryResponse(
            session_id=session_id,
            messages=messages,
            count=len(messages)
        )

    except Exception as e:
        logger.error(f"Error getting messages: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve messages: {str(e)}"
        )


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    session_id: str = Depends(get_session_id),
    manager: ConversationMemoryManager = Depends(get_memory_manager_dependency)
):
    """
    Get memory statistics for session.

    Returns metrics about memory usage.

    **Example Response:**
    ```json
    {
        "total_messages": 42,
        "total_tokens": 1250,
        "strategy": "buffer_window",
        "config": {
            "buffer_window_size": 20,
            "token_limit": 2000,
            "summary_interval": 10
        }
    }
    ```

    Args:
        session_id: Session identifier (from cookie)
        manager: Memory manager instance

    Returns:
        Memory statistics

    Raises:
        HTTPException: If stats retrieval fails
    """
    try:
        stats = await manager.get_statistics()

        logger.info(f"Retrieved stats for session {session_id}: {stats['total_messages']} messages, {stats['total_tokens']} tokens")

        return MemoryStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory statistics: {str(e)}"
        )


# ============================================================================
# MANAGEMENT ENDPOINTS
# ============================================================================

@router.delete("/clear", response_model=ClearMemoryResponse)
async def clear_memory(
    session_id: str = Depends(get_session_id),
    manager: ConversationMemoryManager = Depends(get_memory_manager_dependency)
):
    """
    Clear all memory for current session.

    Permanently deletes all conversation messages for this session.

    **Example Response:**
    ```json
    {
        "message": "Memory cleared successfully",
        "session_id": "abc123-def456"
    }
    ```

    Args:
        session_id: Session identifier (from cookie)
        manager: Memory manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If clear operation fails
    """
    try:
        await manager.clear_memory()

        logger.info(f"Cleared memory for session {session_id}")

        return ClearMemoryResponse(
            message="Memory cleared successfully",
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error clearing memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear memory: {str(e)}"
        )
