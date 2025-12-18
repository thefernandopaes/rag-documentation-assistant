"""
Pytest Configuration and Shared Fixtures

Provides reusable fixtures for:
- FastAPI test client
- Async database sessions
- RAG engine
- Mock data
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DOC_USE_SAMPLE"] = "true"  # Use sample data for faster tests


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def app():
    """
    FastAPI application fixture.

    Scope: module (shared across test module)
    """
    from fastapi_app import app
    return app


@pytest.fixture(scope="module")
def sync_client(app):
    """
    Synchronous test client (FastAPI TestClient).

    Use for simple endpoint testing.
    """
    return TestClient(app)


@pytest.fixture(scope="module")
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """
    Asynchronous test client (httpx AsyncClient).

    Use for concurrent request testing.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async database session fixture.

    Scope: function (new session per test)
    Creates in-memory SQLite database for testing.
    """
    from database_async import get_async_database_uri

    # Use in-memory SQLite for tests
    test_db_uri = "sqlite+aiosqlite:///:memory:"

    engine = create_async_engine(
        test_db_uri,
        echo=False,
        future=True
    )

    # Create tables
    from models_async import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session_factory() as session:
        yield session

    # Cleanup
    await engine.dispose()


@pytest.fixture(scope="module")
def rag_engine():
    """
    RAG engine fixture (async).

    Scope: module (expensive to initialize)
    """
    from rag_engine_async import AsyncRAGEngine
    return AsyncRAGEngine()


@pytest.fixture
def sample_chat_request():
    """Sample chat request data"""
    return {
        "query": "What is FastAPI and how does it compare to Flask?"
    }


@pytest.fixture
def sample_feedback_request():
    """Sample feedback request data"""
    return {
        "conversation_id": "test-conversation-123",
        "feedback": 1
    }


@pytest.fixture
def invalid_queries():
    """Invalid query examples for security testing"""
    return [
        "<script>alert('xss')</script>",
        "SELECT * FROM users WHERE 1=1; --",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "'; DROP TABLE users; --",
    ]


@pytest.fixture
def valid_queries():
    """Valid query examples"""
    return [
        "How do I create a FastAPI endpoint?",
        "What is async/await in Python?",
        "Explain Docker containers",
        "How to use React hooks?",
        "What are AWS Lambda functions?",
    ]


# Pytest markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "concurrent: marks tests for concurrent execution"
    )
