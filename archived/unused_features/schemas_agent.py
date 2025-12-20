"""
Agent Schemas - Pydantic Models for Agent Endpoints

Request/response validation for LangChain agent API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional


# ============================================================================
# AGENT CHAT
# ============================================================================

class AgentChatRequest(BaseModel):
    """Request schema for agent chat endpoint"""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="User query for the agent",
        example="How do I create a FastAPI endpoint with async?"
    )

    use_history: bool = Field(
        default=True,
        description="Whether to use conversation history for context"
    )

    @validator('query')
    def validate_query(cls, v):
        """Validate query format"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")

        # Check for suspicious patterns
        suspicious = ['<script>', 'javascript:', 'onerror=', 'onclick=']
        if any(pattern in v.lower() for pattern in suspicious):
            raise ValueError("Query contains suspicious content")

        return v.strip()


class ToolCall(BaseModel):
    """Information about a tool call made by the agent"""

    tool: str = Field(
        ...,
        description="Tool name",
        example="documentation_search"
    )

    input: Any = Field(
        ...,
        description="Input passed to the tool",
        example="FastAPI async endpoint"
    )

    output: str = Field(
        ...,
        description="Tool output (truncated to 500 chars)",
        example="Found relevant documentation: [1] FastAPI - Async Operations..."
    )


class AgentChatResponse(BaseModel):
    """Response schema for agent chat endpoint"""

    response: str = Field(
        ...,
        description="Agent's final answer",
        example="Here's how to create an async FastAPI endpoint..."
    )

    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tool calls made during reasoning"
    )

    sources: List[str] = Field(
        default_factory=list,
        description="Documentation sources cited",
        example=["FastAPI Documentation - Async Operations", "Python asyncio Guide"]
    )

    session_id: str = Field(
        ...,
        description="Session identifier",
        example="abc123-def456-ghi789"
    )

    response_time: float = Field(
        ...,
        description="Response time in seconds",
        example=2.45
    )


# ============================================================================
# AGENT TOOLS
# ============================================================================

class AgentToolInfo(BaseModel):
    """Information about an available agent tool"""

    name: str = Field(
        ...,
        description="Tool name",
        example="documentation_search"
    )

    description: str = Field(
        ...,
        description="Tool description and usage",
        example="Searches API documentation for relevant information. Use this when..."
    )


# ============================================================================
# AGENT TESTING
# ============================================================================

class AgentTestRequest(BaseModel):
    """Request schema for agent testing endpoint (admin only)"""

    test_queries: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of queries to test",
        example=[
            "What is FastAPI?",
            "How to create an async endpoint?",
            "Show me a Python code example"
        ]
    )

    @validator('test_queries')
    def validate_queries(cls, v):
        """Validate test queries"""
        if not v:
            raise ValueError("Must provide at least one test query")

        # Check each query
        for query in v:
            if not query.strip():
                raise ValueError("Test queries cannot be empty")

            if len(query) > 500:
                raise ValueError("Test queries must be under 500 characters")

        return v


class AgentTestResult(BaseModel):
    """Result for a single test query"""

    query: str = Field(
        ...,
        description="The test query",
        example="What is FastAPI?"
    )

    success: bool = Field(
        ...,
        description="Whether the query succeeded",
        example=True
    )

    response: Optional[str] = Field(
        None,
        description="Agent response (truncated to 200 chars)",
        example="FastAPI is a modern web framework..."
    )

    error: Optional[str] = Field(
        None,
        description="Error message if failed",
        example="Tool execution timeout"
    )

    tool_calls_count: Optional[int] = Field(
        None,
        description="Number of tool calls made",
        example=2
    )

    response_time: float = Field(
        ...,
        description="Response time in seconds",
        example=1.23
    )


class AgentTestSummary(BaseModel):
    """Summary of test results"""

    total: int = Field(
        ...,
        description="Total number of queries tested",
        example=5
    )

    successful: int = Field(
        ...,
        description="Number of successful queries",
        example=4
    )

    failed: int = Field(
        ...,
        description="Number of failed queries",
        example=1
    )

    avg_response_time: float = Field(
        ...,
        description="Average response time in seconds",
        example=1.45
    )


class AgentTestResponse(BaseModel):
    """Response schema for agent testing endpoint"""

    results: List[AgentTestResult] = Field(
        ...,
        description="Individual test results"
    )

    summary: AgentTestSummary = Field(
        ...,
        description="Summary statistics"
    )
