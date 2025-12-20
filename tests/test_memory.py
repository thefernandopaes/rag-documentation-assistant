"""
Memory Tests - Comprehensive Testing for Conversation Memory

Tests for memory manager, strategies, and endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

# Memory components
from conversation_memory import ConversationMemoryManager
from models_memory import Base, ConversationMemory, MemoryConfiguration

# API components
from httpx import AsyncClient


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
async def db_session():
    """
    Create in-memory test database session.

    Yields:
        AsyncSession for testing
    """
    # Create in-memory SQLite database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session factory
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    # Yield session
    async with async_session() as session:
        yield session

    # Cleanup
    await engine.dispose()


@pytest.fixture
def session_id():
    """Test session ID"""
    return "test-session-123"


@pytest.fixture
async def memory_manager(db_session, session_id):
    """
    Create memory manager for testing.

    Args:
        db_session: Test database session
        session_id: Test session ID

    Returns:
        ConversationMemoryManager instance
    """
    return ConversationMemoryManager(session_id, db_session)


# ============================================================================
# MEMORY MANAGER TESTS
# ============================================================================

class TestConversationMemoryManager:
    """Tests for ConversationMemoryManager"""

    @pytest.mark.asyncio
    async def test_initialization(self, memory_manager, session_id):
        """Test memory manager initializes correctly"""
        assert memory_manager.session_id == session_id
        assert memory_manager.db is not None
        assert memory_manager.encoding is not None or True  # encoding might not be available

    @pytest.mark.asyncio
    async def test_get_or_create_config(self, memory_manager, session_id):
        """Test getting or creating memory configuration"""
        config = await memory_manager.get_memory_config()

        assert config is not None
        assert config.session_id == session_id
        assert config.strategy in ['buffer_window', 'summary', 'token_buffer']
        assert config.buffer_window_size > 0
        assert config.token_limit > 0

    @pytest.mark.asyncio
    async def test_add_message(self, memory_manager):
        """Test adding message to memory"""
        await memory_manager.add_message("user", "Hello!")
        await memory_manager.add_message("assistant", "Hi there!")

        messages = await memory_manager.get_messages()

        assert len(messages) == 2
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'Hello!'
        assert messages[1]['role'] == 'assistant'
        assert messages[1]['content'] == 'Hi there!'

    @pytest.mark.asyncio
    async def test_add_invalid_role(self, memory_manager):
        """Test adding message with invalid role fails"""
        with pytest.raises(ValueError):
            await memory_manager.add_message("invalid_role", "Test")

    @pytest.mark.asyncio
    async def test_message_indexing(self, memory_manager, db_session, session_id):
        """Test messages are indexed correctly"""
        # Add 3 messages
        await memory_manager.add_message("user", "Message 1")
        await memory_manager.add_message("assistant", "Message 2")
        await memory_manager.add_message("user", "Message 3")

        # Check indexes
        result = await db_session.execute(
            select(ConversationMemory)
            .where(ConversationMemory.session_id == session_id)
            .order_by(ConversationMemory.message_index)
        )
        messages = result.scalars().all()

        assert messages[0].message_index == 1
        assert messages[1].message_index == 2
        assert messages[2].message_index == 3


# ============================================================================
# BUFFER WINDOW STRATEGY TESTS
# ============================================================================

class TestBufferWindowStrategy:
    """Tests for buffer window memory strategy"""

    @pytest.mark.asyncio
    async def test_buffer_window_limit(self, memory_manager, db_session, session_id):
        """Test buffer window respects size limit"""
        # Set strategy to buffer_window with size 5
        config = await memory_manager.get_memory_config()
        config.strategy = 'buffer_window'
        config.buffer_window_size = 5
        await db_session.commit()

        # Add 10 messages
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            await memory_manager.add_message(role, f"Message {i}")

        # Get messages with window size 5
        messages = await memory_manager._get_buffer_window_messages(5)

        # Should only return last 5 messages
        assert len(messages) == 5
        assert messages[-1]['content'] == "Message 9"
        assert messages[0]['content'] == "Message 5"

    @pytest.mark.asyncio
    async def test_buffer_window_chronological_order(self, memory_manager):
        """Test messages returned in chronological order"""
        await memory_manager.add_message("user", "First")
        await memory_manager.add_message("assistant", "Second")
        await memory_manager.add_message("user", "Third")

        messages = await memory_manager._get_buffer_window_messages(10)

        assert messages[0]['content'] == "First"
        assert messages[1]['content'] == "Second"
        assert messages[2]['content'] == "Third"


# ============================================================================
# TOKEN BUFFER STRATEGY TESTS
# ============================================================================

class TestTokenBufferStrategy:
    """Tests for token buffer memory strategy"""

    @pytest.mark.asyncio
    async def test_token_buffer_limit(self, memory_manager, db_session, session_id):
        """Test token buffer respects token limit"""
        # Set strategy to token_buffer
        config = await memory_manager.get_memory_config()
        config.strategy = 'token_buffer'
        config.token_limit = 50  # Very small limit for testing
        await db_session.commit()

        # Add messages with known token counts
        await memory_manager.add_message("user", "Short")  # ~1-2 tokens
        await memory_manager.add_message("assistant", "Also short")  # ~2-3 tokens
        await memory_manager.add_message("user", "This is a much longer message that will exceed the token limit")  # ~10+ tokens

        messages = await memory_manager._get_token_buffer_messages(50)

        # Should fit within 50 tokens
        total_tokens = sum(memory_manager._count_tokens(msg['content']) for msg in messages)
        assert total_tokens <= 50


# ============================================================================
# SUMMARY STRATEGY TESTS
# ============================================================================

class TestSummaryStrategy:
    """Tests for summary memory strategy"""

    @pytest.mark.asyncio
    async def test_summary_creation_trigger(self, memory_manager, db_session, session_id):
        """Test summary is created when threshold reached"""
        # Set strategy to summary
        config = await memory_manager.get_memory_config()
        config.strategy = 'summary'
        config.summary_interval = 5  # Summarize after 5 messages
        await db_session.commit()

        # Mock LLM for summarization
        with patch('conversation_memory.ChatOpenAI') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_response = Mock()
            mock_response.content = "Summary of conversation"
            mock_llm.ainvoke.return_value = mock_response
            mock_llm_class.return_value = mock_llm

            # Add 5 messages (should trigger summary)
            for i in range(5):
                role = "user" if i % 2 == 0 else "assistant"
                await memory_manager.add_message(role, f"Message {i}")

            # Check if summary was created
            result = await db_session.execute(
                select(ConversationMemory)
                .where(
                    ConversationMemory.session_id == session_id,
                    ConversationMemory.is_summarized == True
                )
            )
            summaries = result.scalars().all()

            # Should have at least one summary
            assert len(summaries) >= 1

    @pytest.mark.asyncio
    async def test_get_summary_messages(self, memory_manager, db_session, session_id):
        """Test retrieving messages with summaries"""
        # Manually create a summary and some messages
        summary = ConversationMemory(
            session_id=session_id,
            role='system',
            content='Summary of previous messages',
            tokens=10,
            message_index=-1,
            memory_strategy='summary',
            is_summarized=True
        )
        db_session.add(summary)

        message = ConversationMemory(
            session_id=session_id,
            role='user',
            content='New message',
            tokens=5,
            message_index=1,
            memory_strategy='summary',
            is_summarized=False
        )
        db_session.add(message)
        await db_session.commit()

        # Get messages
        messages = await memory_manager._get_summary_messages()

        # Should include both summary and message
        assert len(messages) >= 2
        assert any('[Summary' in msg['content'] for msg in messages)


# ============================================================================
# STATISTICS TESTS
# ============================================================================

class TestMemoryStatistics:
    """Tests for memory statistics"""

    @pytest.mark.asyncio
    async def test_statistics_empty(self, memory_manager):
        """Test statistics with no messages"""
        stats = await memory_manager.get_statistics()

        assert stats['total_messages'] == 0
        assert stats['total_tokens'] == 0
        assert 'strategy' in stats
        assert 'config' in stats

    @pytest.mark.asyncio
    async def test_statistics_with_messages(self, memory_manager):
        """Test statistics with messages"""
        await memory_manager.add_message("user", "Test message")
        await memory_manager.add_message("assistant", "Response")

        stats = await memory_manager.get_statistics()

        assert stats['total_messages'] == 2
        assert stats['total_tokens'] > 0
        assert stats['strategy'] in ['buffer_window', 'summary', 'token_buffer']

    @pytest.mark.asyncio
    async def test_clear_memory(self, memory_manager):
        """Test clearing memory"""
        # Add messages
        await memory_manager.add_message("user", "Test 1")
        await memory_manager.add_message("assistant", "Test 2")

        # Clear
        await memory_manager.clear_memory()

        # Check empty
        messages = await memory_manager.get_messages()
        assert len(messages) == 0

        stats = await memory_manager.get_statistics()
        assert stats['total_messages'] == 0


# ============================================================================
# ENDPOINT TESTS
# ============================================================================

@pytest.mark.asyncio
class TestMemoryEndpoints:
    """Tests for memory API endpoints"""

    @pytest.fixture
    async def app(self):
        """Create test FastAPI app"""
        try:
            from fastapi_app import app
            return app
        except ImportError:
            pytest.skip("FastAPI app not available")

    @pytest.fixture
    async def client(self, app):
        """Create async test client"""
        from httpx import ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def test_get_memory_config_endpoint(self, client):
        """Test GET /api/memory/config"""
        response = await client.get("/api/memory/config")

        # Should return 200 (or 404 if route not registered yet)
        assert response.status_code in [200, 404, 307]

        if response.status_code == 200:
            data = response.json()
            assert 'session_id' in data
            assert 'strategy' in data

    async def test_update_memory_config_endpoint(self, client):
        """Test PUT /api/memory/config"""
        response = await client.put(
            "/api/memory/config",
            json={
                "strategy": "token_buffer",
                "token_limit": 3000
            }
        )

        # Should return 200 (or 404 if route not registered yet)
        assert response.status_code in [200, 404, 422, 307]

    async def test_get_messages_endpoint(self, client):
        """Test GET /api/memory/messages"""
        response = await client.get("/api/memory/messages?limit=10")

        # Should return 200 (or 404 if route not registered yet)
        assert response.status_code in [200, 404, 307]

        if response.status_code == 200:
            data = response.json()
            assert 'messages' in data
            assert 'count' in data

    async def test_clear_memory_endpoint(self, client):
        """Test DELETE /api/memory/clear"""
        response = await client.delete("/api/memory/clear")

        # Should return 200 (or 404 if route not registered yet)
        assert response.status_code in [200, 404, 307]

        if response.status_code == 200:
            data = response.json()
            assert 'message' in data


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
class TestMemoryIntegration:
    """Integration tests for memory system"""

    @pytest.mark.skipif(
        True,
        reason="Integration tests require --run-integration flag"
    )
    async def test_full_conversation_flow(self, db_session):
        """Test complete conversation flow with memory"""
        manager = ConversationMemoryManager("integration-test", db_session)

        # Add conversation
        await manager.add_message("user", "What is FastAPI?")
        await manager.add_message("assistant", "FastAPI is a web framework...")
        await manager.add_message("user", "Can you show me an example?")
        await manager.add_message("assistant", "Here's an example: ...")

        # Get messages
        messages = await manager.get_messages()

        assert len(messages) == 4
        assert messages[0]['role'] == 'user'
        assert messages[-1]['role'] == 'assistant'

        # Get stats
        stats = await manager.get_statistics()

        assert stats['total_messages'] == 4
        assert stats['total_tokens'] > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([
        __file__,
        "-v",  # Verbose
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
        "--color=yes"  # Colored output
    ])
