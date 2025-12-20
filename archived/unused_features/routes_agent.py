"""
Agent Routes - FastAPI Endpoints for LangChain Agent

Provides agent-powered chat endpoints with multi-tool orchestration.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
import logging
import time

from database_async import get_db
from langchain_agent import get_agent, DocumentationAgent
from schemas_agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentToolInfo,
    AgentTestRequest,
    AgentTestResponse
)
from models_async import Conversation
from config import Config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


# ============================================================================
# DEPENDENCIES
# ============================================================================

async def get_session_id(request: Request) -> str:
    """
    Get or create session ID from cookies.

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


async def validate_admin_key(request: Request):
    """
    Validate admin API key for protected endpoints.

    Args:
        request: FastAPI request

    Raises:
        HTTPException: If authentication fails
    """
    if not Config._is_production():
        # Skip auth in development
        return

    api_key = request.headers.get("X-Admin-Key")
    if not api_key or api_key != Config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing admin API key"
        )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    session_id: str = Depends(get_session_id),
    db: AsyncSession = Depends(get_db),
    agent: DocumentationAgent = Depends(get_agent)
):
    """
    Agent-powered chat endpoint.

    Uses LangChain ReAct agent with multiple tools:
    - documentation_search: Search API docs via RAG
    - code_generator: Generate code examples
    - code_validator: Validate generated code

    **Example Request:**
    ```json
    {
        "query": "Show me how to create a FastAPI endpoint with async",
        "use_history": true
    }
    ```

    **Example Response:**
    ```json
    {
        "response": "Here's how to create an async FastAPI endpoint...",
        "tool_calls": [
            {
                "tool": "documentation_search",
                "input": "FastAPI async endpoint",
                "output": "Found relevant documentation..."
            },
            {
                "tool": "code_generator",
                "input": "Python FastAPI async POST endpoint",
                "output": "```python\\nfrom fastapi import FastAPI..."
            }
        ],
        "sources": ["FastAPI Documentation - Async Operations"],
        "session_id": "abc123",
        "response_time": 2.45
    }
    ```

    Args:
        request: Agent chat request
        session_id: User session ID (from cookie)
        db: Database session
        agent: LangChain agent instance

    Returns:
        Agent response with tool calls and sources

    Raises:
        HTTPException: If query validation fails or agent errors
    """
    start_time = time.time()

    try:
        logger.info(f"Agent chat request: {request.query[:100]}...")

        # Get conversation history if requested
        conversation_history = None
        if request.use_history:
            conversation_history = await _get_conversation_history(db, session_id)

        # Run agent
        result = await agent.arun(
            query=request.query,
            conversation_history=conversation_history
        )

        # Extract sources from tool calls
        sources = _extract_sources_from_tool_calls(result['tool_calls'])

        # Save conversation to database
        conversation = Conversation(
            session_id=session_id,
            user_query=request.query,
            bot_response=result['output'],
            response_time=time.time() - start_time,
            sources=",".join(sources) if sources else None,
            model_used=Config.AGENT_MODEL if hasattr(Config, 'AGENT_MODEL') else "gpt-4"
        )
        db.add(conversation)
        await db.commit()

        logger.info(f"Agent response generated in {time.time() - start_time:.2f}s")

        return AgentChatResponse(
            response=result['output'],
            tool_calls=result['tool_calls'],
            sources=sources,
            session_id=session_id,
            response_time=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"Agent chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}"
        )


@router.get("/tools", response_model=List[AgentToolInfo])
async def list_agent_tools(
    agent: DocumentationAgent = Depends(get_agent)
):
    """
    List available agent tools.

    Returns information about all tools the agent can use.

    **Example Response:**
    ```json
    [
        {
            "name": "documentation_search",
            "description": "Searches API documentation for relevant information..."
        },
        {
            "name": "code_generator",
            "description": "Generates code examples in various programming languages..."
        },
        {
            "name": "code_validator",
            "description": "Validates code syntax and checks for basic errors..."
        }
    ]
    ```

    Args:
        agent: LangChain agent instance

    Returns:
        List of available tools with descriptions
    """
    try:
        tools_info = agent.get_tools_info()
        return [
            AgentToolInfo(name=tool['name'], description=tool['description'])
            for tool in tools_info
        ]

    except Exception as e:
        logger.error(f"Error listing tools: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tools: {str(e)}"
        )


@router.post("/test", response_model=AgentTestResponse, dependencies=[Depends(validate_admin_key)])
async def test_agent(
    request: AgentTestRequest,
    agent: DocumentationAgent = Depends(get_agent)
):
    """
    Test agent with predefined queries (admin only).

    Requires admin API key in X-Admin-Key header.

    **Example Request:**
    ```json
    {
        "test_queries": [
            "What is FastAPI?",
            "How to create an async endpoint?",
            "Show me a Python code example"
        ]
    }
    ```

    **Example Response:**
    ```json
    {
        "results": [
            {
                "query": "What is FastAPI?",
                "success": true,
                "response": "FastAPI is a modern web framework...",
                "tool_calls_count": 1,
                "response_time": 1.23
            },
            ...
        ],
        "summary": {
            "total": 3,
            "successful": 3,
            "failed": 0,
            "avg_response_time": 1.45
        }
    }
    ```

    Args:
        request: Test request with queries
        agent: LangChain agent instance

    Returns:
        Test results with summary

    Raises:
        HTTPException: If authentication fails
    """
    try:
        logger.info(f"Testing agent with {len(request.test_queries)} queries")

        results = []
        total_time = 0.0

        for query in request.test_queries:
            start_time = time.time()

            try:
                # Run agent
                result = await agent.arun(query=query)

                response_time = time.time() - start_time
                total_time += response_time

                results.append({
                    'query': query,
                    'success': True,
                    'response': result['output'][:200] + "..." if len(result['output']) > 200 else result['output'],
                    'tool_calls_count': len(result['tool_calls']),
                    'response_time': response_time
                })

            except Exception as e:
                logger.error(f"Test query failed: {query} - {e}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'response_time': time.time() - start_time
                })

        # Calculate summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        avg_time = total_time / len(results) if results else 0.0

        logger.info(f"Agent test completed: {successful}/{len(results)} successful")

        return AgentTestResponse(
            results=results,
            summary={
                'total': len(results),
                'successful': successful,
                'failed': failed,
                'avg_response_time': avg_time
            }
        )

    except Exception as e:
        logger.error(f"Agent test error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent test failed: {str(e)}"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _get_conversation_history(
    db: AsyncSession,
    session_id: str,
    limit: int = 5
) -> List[Dict[str, str]]:
    """
    Get conversation history for session.

    Args:
        db: Database session
        session_id: User session ID
        limit: Max number of messages to retrieve

    Returns:
        List of conversation messages
    """
    from sqlalchemy import select

    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .order_by(Conversation.timestamp.desc())
        .limit(limit)
    )
    conversations = result.scalars().all()

    # Format as chat history (newest first, so reverse)
    history = []
    for conv in reversed(conversations):
        history.append({'role': 'user', 'content': conv.user_query})
        history.append({'role': 'assistant', 'content': conv.bot_response})

    return history


def _extract_sources_from_tool_calls(tool_calls: List[Dict]) -> List[str]:
    """
    Extract source URLs/titles from tool calls.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        List of unique sources
    """
    sources = set()

    for call in tool_calls:
        # Look for documentation_search results
        if call.get('tool') == 'documentation_search':
            output = call.get('output', '')

            # Extract [1], [2], etc. sources
            import re
            matches = re.findall(r'\[(\d+)\]\s+([^\n]+)', output)
            for _, title in matches:
                sources.add(title.strip())

    return list(sources)
