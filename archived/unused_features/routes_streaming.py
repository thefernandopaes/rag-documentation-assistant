"""
Streaming Routes - FastAPI Endpoints for Streaming Responses

Provides Server-Sent Events (SSE) streaming for real-time agent responses.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import logging

from langchain_agent import get_agent, DocumentationAgent
from streaming.stream_handler import stream_agent_response, stream_text_response
from schemas_agent import AgentChatRequest
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stream", tags=["Streaming"])


# ============================================================================
# SCHEMAS
# ============================================================================

class StreamRequest(BaseModel):
    """Request schema for streaming endpoint"""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="User query to stream response for",
        example="What is FastAPI and how does it work?"
    )


# ============================================================================
# DEPENDENCIES
# ============================================================================

async def get_session_id(request: Request) -> str:
    """
    Get session ID from cookie.

    Args:
        request: FastAPI request

    Returns:
        Session ID
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    return session_id


# ============================================================================
# STREAMING ENDPOINTS
# ============================================================================

@router.post("/chat")
async def stream_chat(
    request: StreamRequest,
    session_id: str = Depends(get_session_id),
    agent: DocumentationAgent = Depends(get_agent)
):
    """
    Stream agent response in real-time using Server-Sent Events (SSE).

    Returns a continuous stream of tokens as the agent generates the response.

    **SSE Event Format:**
    ```
    data: {"token": "Hello"}
    data: {"token": " world"}
    data: {"done": true, "sources": ["..."]}
    ```

    **Client Usage (JavaScript):**
    ```javascript
    const response = await fetch('/api/stream/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: "What is FastAPI?"})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.token) console.log(data.token);
                if (data.done) break;
            }
        }
    }
    ```

    **cURL Usage:**
    ```bash
    curl -N -X POST http://localhost:8000/api/stream/chat \\
      -H "Content-Type: application/json" \\
      -d '{"query": "What is FastAPI?"}'
    ```

    Args:
        request: Stream request with query
        session_id: Session identifier (from cookie)
        agent: Documentation agent instance

    Returns:
        StreamingResponse with SSE content

    Raises:
        HTTPException: If streaming fails
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Generate SSE events.

        Yields:
            SSE-formatted event strings
        """
        try:
            logger.info(f"Starting stream for session {session_id}: {request.query[:50]}...")

            # Stream agent response
            async for event in stream_agent_response(
                agent=agent,
                query=request.query,
                session_id=session_id
            ):
                yield event

            logger.info(f"Stream completed for session {session_id}")

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)

            # Send error event
            error_event = f"data: {json.dumps({'error': str(e)})}\n\n"
            yield error_event

            # Send done event
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",  # CORS for frontend
        }
    )


@router.get("/test")
async def stream_test():
    """
    Test streaming endpoint with predefined text.

    Useful for testing SSE setup without calling the agent/LLM.

    **Example:**
    ```bash
    curl -N http://localhost:8000/api/stream/test
    ```

    Returns:
        StreamingResponse with test content
    """
    test_text = (
        "This is a test of the streaming system. "
        "Each chunk is sent with a small delay to simulate real streaming. "
        "This endpoint is useful for testing SSE without making LLM calls."
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate test events"""
        try:
            async for event in stream_text_response(test_text, chunk_size=15):
                yield event

        except Exception as e:
            logger.error(f"Test streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/health")
async def stream_health():
    """
    Health check for streaming endpoints.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "streaming": "enabled",
        "endpoints": [
            "/api/stream/chat",
            "/api/stream/test"
        ]
    }
