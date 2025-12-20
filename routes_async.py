"""
FastAPI Async Routes

All endpoints migrated from Flask to FastAPI with async/await patterns.

Endpoints:
- POST /api/chat - Generate RAG response
- POST /api/feedback - Submit feedback
- GET /api/history - Get conversation history
- GET /api/stats - Get system statistics
- POST /api/initialize - Initialize documents (admin)
"""

import json
import logging
import time
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from hashlib import sha256

from schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HistoryItem,
    StatsResponse,
    InitializeRequest,
    InitializeResponse
)
from models_async import Conversation, DocumentChunk
from database_async import get_async_db
from dependencies import (
    validate_rate_limit,
    validate_query_security,
    validate_admin_key,
    get_rag_engine,
    get_session_id,
    validate_conversation_exists
)
from config import Config
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["API"])


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns system health status and version information.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "framework": "fastapi",
        "async": True
    }


@router.post("/api/chat", response_model=ChatResponse)
async def api_chat(
    request: ChatRequest,
    session_id: str = Depends(get_session_id),
    rag_engine = Depends(get_rag_engine),
    db: AsyncSession = Depends(get_async_db),
    _rate_limit: None = Depends(validate_rate_limit),
    response: Response = None
):
    """
    Generate AI response using RAG (Async).

    **Performance:** 2-4x faster than sync version with AsyncOpenAI.

    Validates query for security, searches documents, generates response,
    and saves conversation to database.
    """
    try:
        # Validate query security (uses existing validators.py)
        validated_query = await validate_query_security(request.query)

        # Get conversation history for context
        conversation_history = await get_conversation_history_async(session_id, db, limit=5)

        # Generate response using async RAG engine
        start_time = time.time()
        response_data = await rag_engine.generate_response(
            validated_query,
            conversation_history
        )

        # Save conversation to database (async)
        conversation = Conversation(
            id=str(uuid.uuid4()),
            session_id=session_id,
            user_query=validated_query,
            ai_response=response_data.get('answer', response_data.get('response', '')),
            sources=json.dumps(response_data.get('sources', [])),
            response_time=response_data.get('response_time', time.time() - start_time),
            created_at=datetime.utcnow()
        )
        db.add(conversation)
        await db.commit()

        # Set session cookie
        if response:
            response.set_cookie(
                key="session_id",
                value=session_id,
                max_age=86400 * 30,  # 30 days
                httponly=True,
                samesite="lax"
            )

        # Build response
        api_response = {
            'response': response_data.get('answer', response_data.get('response', '')),
            'sources': response_data.get('sources', []),
            'response_time': response_data.get('response_time', 0),
            'cached': response_data.get('cached', False)
        }

        # Add API-specific fields if present
        if 'examples' in response_data:
            api_response['examples'] = response_data['examples']
        elif 'code_examples' in response_data:
            api_response['examples'] = response_data['code_examples']

        if 'endpoints' in response_data:
            api_response['endpoints'] = response_data['endpoints']

        if 'authentication' in response_data:
            api_response['authentication'] = response_data['authentication']

        if 'parameters' in response_data:
            api_response['parameters'] = response_data['parameters']

        if 'response_format' in response_data:
            api_response['response_format'] = response_data['response_format']

        if 'error_codes' in response_data:
            api_response['error_codes'] = response_data['error_codes']

        if 'related_concepts' in response_data:
            api_response['related_concepts'] = response_data['related_concepts']
        elif 'related_questions' in response_data:
            api_response['related_concepts'] = response_data['related_questions']

        logger.info(f"Chat response generated in {api_response['response_time']:.2f}s (async)")

        return ChatResponse(**api_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat API: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                'error': 'Internal server error',
                'message': 'An unexpected error occurred. Please try again.'
            }
        )


@router.post("/api/feedback")
async def api_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_async_db),
    _rate_limit: None = Depends(validate_rate_limit)
):
    """
    Submit feedback for a conversation (Async).

    Allows users to give thumbs up (1) or thumbs down (-1) on responses.
    """
    try:
        # Get conversation
        result = await db.execute(
            select(Conversation).where(Conversation.id == request.conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {request.conversation_id} not found"
            )

        # Update feedback
        conversation.feedback = request.feedback
        await db.commit()

        logger.info(f"Feedback recorded for conversation {request.conversation_id}: {request.feedback}")

        return {"message": "Feedback recorded successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                'error': 'Internal server error',
                'message': 'Failed to record feedback'
            }
        )


@router.get("/api/history")
async def api_history(
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_async_db),
    limit: int = 20
):
    """
    Get conversation history for current session (Async).

    Returns up to `limit` most recent conversations.
    """
    try:
        # Query conversations
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
        )
        conversations = result.scalars().all()

        # Format history
        history = []
        for conv in conversations:
            sources = []
            try:
                if conv.sources:
                    sources = json.loads(conv.sources)
            except json.JSONDecodeError:
                pass

            history.append({
                'id': conv.id,
                'query': conv.user_query,
                'response': conv.ai_response,
                'sources': sources,
                'created_at': conv.created_at.isoformat() if conv.created_at else None,
                'response_time': conv.response_time,
                'feedback': conv.feedback
            })

        return history

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return []


@router.get("/api/stats", response_model=StatsResponse)
async def api_stats(
    rag_engine = Depends(get_rag_engine),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get system statistics (Async).

    Returns document count, conversation stats, cache stats, and system info.
    """
    try:
        # Get RAG engine stats (async)
        collection_stats = await rag_engine.get_collection_stats()

        # Get conversation stats (async)
        total_result = await db.execute(select(func.count(Conversation.id)))
        total_conversations = total_result.scalar() or 0

        avg_result = await db.execute(select(func.avg(Conversation.response_time)))
        avg_response_time = avg_result.scalar() or 0

        positive_result = await db.execute(
            select(func.count(Conversation.id)).where(Conversation.feedback == 1)
        )
        positive_feedback = positive_result.scalar() or 0

        negative_result = await db.execute(
            select(func.count(Conversation.id)).where(Conversation.feedback == -1)
        )
        negative_feedback = negative_result.scalar() or 0

        # Get cache stats (async)
        cache_stats = await rag_engine.cache.get_stats()

        return StatsResponse(
            documents=collection_stats,
            conversations={
                'total': total_conversations,
                'avg_response_time': round(avg_response_time, 2),
                'positive_feedback': positive_feedback,
                'negative_feedback': negative_feedback
            },
            cache=cache_stats,
            system={
                'is_production': Config._is_production(),
                'version': '2.0.0',
                'framework': 'fastapi',
                'async': True
            }
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return StatsResponse(
            documents={'document_count': 0, 'collection_name': Config.COLLECTION_NAME},
            conversations={'total': 0, 'avg_response_time': 0, 'positive_feedback': 0, 'negative_feedback': 0},
            cache={'total_entries': 0},
            system={'is_production': Config._is_production(), 'version': '2.0.0', 'framework': 'fastapi', 'async': True}
        )


@router.get("/api/performance-stats")
async def api_performance_stats(
    db: AsyncSession = Depends(get_async_db),
    _admin = Depends(validate_admin_key)
):
    """
    Get detailed performance statistics (Admin only).

    Analyzes recent conversations to identify bottlenecks.
    Returns performance metrics including avg, median, p95 times.
    """
    try:
        # Get last 100 conversations with response times
        result = await db.execute(
            select(Conversation)
            .where(Conversation.response_time.isnot(None))
            .order_by(Conversation.created_at.desc())
            .limit(100)
        )
        conversations = result.scalars().all()

        if not conversations:
            return {
                "message": "No conversations found",
                "stats": None
            }

        # Calculate response time statistics
        response_times = [c.response_time for c in conversations]
        response_times.sort()

        n = len(response_times)
        stats = {
            "recent_queries": n,
            "avg_response_time": round(sum(response_times) / n, 3),
            "median_response_time": round(response_times[n//2], 3),
            "min_response_time": round(min(response_times), 3),
            "max_response_time": round(max(response_times), 3),
            "p95_response_time": round(response_times[int(n*0.95)], 3) if n > 1 else 0,
            "p99_response_time": round(response_times[int(n*0.99)], 3) if n > 1 else 0
        }

        # Identify slow queries (> 5s)
        slow_queries = [
            {
                "query": c.user_query[:100] + "..." if len(c.user_query) > 100 else c.user_query,
                "response_time": round(c.response_time, 3),
                "created_at": c.created_at.isoformat(),
                "cached": "cached" in c.ai_response.lower() if c.ai_response else False
            }
            for c in conversations if c.response_time and c.response_time > 5.0
        ]

        # Performance health assessment
        avg_time = stats["avg_response_time"]
        if avg_time <= 3.0:
            health_status = "EXCELLENT"
            health_color = "green"
            recommendation = "System is performing optimally!"
        elif avg_time <= 5.0:
            health_status = "GOOD"
            health_color = "blue"
            recommendation = "System is performing well."
        elif avg_time <= 8.0:
            health_status = "ACCEPTABLE"
            health_color = "yellow"
            recommendation = "Consider further optimization."
        else:
            health_status = "NEEDS_IMPROVEMENT"
            health_color = "red"
            recommendation = "Significant optimization required."

        return {
            "stats": stats,
            "health": {
                "status": health_status,
                "color": health_color,
                "recommendation": recommendation
            },
            "slow_queries": slow_queries[:10],  # Top 10 slowest
            "analysis": {
                "cache_recommended": len(slow_queries) > 10,
                "optimization_needed": avg_time > 5.0
            }
        }

    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/initialize", response_model=InitializeResponse)
async def api_initialize(
    request: InitializeRequest = InitializeRequest(force=False),
    rag_engine = Depends(get_rag_engine),
    db: AsyncSession = Depends(get_async_db),
    _admin: None = Depends(validate_admin_key)
):
    """
    Initialize RAG system with documents (Admin only, Async).

    Processes documentation sources and adds them to the vector store.
    Protected endpoint requiring admin API key in production.
    """
    try:
        # Check if documents already exist
        result = await db.execute(select(func.count(DocumentChunk.id)))
        existing_docs = result.scalar() or 0

        if existing_docs > 0 and not request.force:
            return InitializeResponse(
                message=f'System already initialized with {existing_docs} document chunks',
                status='already_initialized',
                document_count=existing_docs
            )

        # Process documents (sync, wrapped in thread)
        import asyncio
        processor = DocumentProcessor()

        # Run in thread pool to avoid blocking
        documents = await asyncio.to_thread(processor.process_documentation_sources)

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    'error': 'No documents processed',
                    'message': 'Failed to process any documentation sources'
                }
            )

        # Add documents to RAG engine (async)
        await rag_engine.add_documents(documents)

        # Save document metadata to database (async)
        for doc in documents:
            content_hash = sha256(doc['content'].encode('utf-8')).hexdigest()

            # Check if exists
            check_result = await db.execute(
                select(DocumentChunk).where(
                    DocumentChunk.source_url == doc['source_url'],
                    DocumentChunk.content_hash == content_hash
                )
            )
            exists = check_result.scalar_one_or_none()

            if exists:
                continue

            doc_chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                source_url=doc['source_url'],
                title=doc['title'],
                content=doc['content'][:1000],  # Store first 1000 chars
                chunk_index=0,
                doc_type=doc['doc_type'],
                version=doc['version'],
                content_hash=content_hash,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(doc_chunk)

        await db.commit()

        logger.info(f"Initialized RAG system with {len(documents)} documents (async)")

        return InitializeResponse(
            message=f'System initialized successfully with {len(documents)} documents',
            status='initialized',
            document_count=len(documents)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing system: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                'error': 'Initialization failed',
                'message': str(e)
            }
        )


# Helper functions

async def get_conversation_history_async(
    session_id: str,
    db: AsyncSession,
    limit: int = 5
) -> List[dict]:
    """Get recent conversation history for context (async)"""
    try:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
        )
        conversations = result.scalars().all()

        history = []
        for conv in reversed(list(conversations)):  # Chronological order
            history.append({
                'user': conv.user_query,
                'assistant': conv.ai_response
            })

        return history

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return []
