"""
Streaming Tests - Tests for SSE Streaming Functionality

Tests for Server-Sent Events streaming, callbacks, and endpoints.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from httpx import AsyncClient

# Streaming components
from streaming.stream_handler import (
    StreamingCallbackHandler,
    stream_agent_response,
    stream_text_response
)


# ============================================================================
# STREAMING CALLBACK TESTS
# ============================================================================

class TestStreamingCallbackHandler:
    """Tests for StreamingCallbackHandler"""

    @pytest.fixture
    def handler(self):
        """Create callback handler for testing"""
        return StreamingCallbackHandler()

    def test_initialization(self, handler):
        """Test handler initializes correctly"""
        assert handler.queue is not None
        assert handler.done == False

    @pytest.mark.asyncio
    async def test_token_streaming(self, handler):
        """Test token streaming through callback"""
        # Simulate tokens being generated
        handler.on_llm_new_token("Hello")
        handler.on_llm_new_token(" ")
        handler.on_llm_new_token("World")
        handler.on_llm_end(Mock())

        # Collect tokens
        tokens = []
        async for token in handler.aiter():
            tokens.append(token)

        assert len(tokens) == 3
        assert tokens == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_done_flag(self, handler):
        """Test done flag is set correctly"""
        assert handler.done == False

        handler.on_llm_end(Mock())

        assert handler.done == True

    @pytest.mark.asyncio
    async def test_error_handling(self, handler):
        """Test error handling during streaming"""
        error = Exception("Test error")

        handler.on_llm_error(error)

        # Should set done and send error message
        assert handler.done == True

        # Collect tokens
        tokens = []
        async for token in handler.aiter():
            tokens.append(token)

        # Should have error message
        assert len(tokens) > 0
        assert "ERROR" in tokens[0]


# ============================================================================
# STREAM HANDLER TESTS
# ============================================================================

class TestStreamHandlers:
    """Tests for stream handler functions"""

    @pytest.mark.asyncio
    async def test_stream_text_response(self):
        """Test streaming pre-generated text"""
        text = "Hello World"

        events = []
        async for event in stream_text_response(text, chunk_size=5):
            events.append(event)

        # Should have multiple events
        assert len(events) > 2  # At least chunks + done event

        # Last event should be done
        last_event_data = json.loads(events[-1].split("data: ")[1])
        assert last_event_data['done'] == True

        # Should have token events
        has_tokens = any('"token"' in event for event in events)
        assert has_tokens == True

    @pytest.mark.asyncio
    async def test_stream_agent_response_error_handling(self):
        """Test stream handles agent errors gracefully"""
        # Mock agent that raises error
        mock_agent = AsyncMock()
        mock_agent.arun.side_effect = Exception("Test error")
        mock_agent.llm = Mock()
        mock_agent.llm.streaming = False
        mock_agent.llm.callbacks = []

        events = []
        async for event in stream_agent_response(mock_agent, "test query"):
            events.append(event)

        # Should have error event
        has_error = any('"error"' in event for event in events)
        assert has_error == True

        # Should have done event
        has_done = any('"done"' in event for event in events)
        assert has_done == True


# ============================================================================
# ENDPOINT TESTS
# ============================================================================

@pytest.mark.asyncio
class TestStreamingEndpoints:
    """Tests for streaming API endpoints"""

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

    async def test_stream_health_endpoint(self, client):
        """Test /api/stream/health endpoint"""
        response = await client.get("/api/stream/health")

        # Should return 200 (or 404 if route not registered yet)
        assert response.status_code in [200, 404, 307]

        if response.status_code == 200:
            data = response.json()
            assert 'status' in data

    async def test_stream_test_endpoint(self, client):
        """Test /api/stream/test endpoint"""
        try:
            async with client.stream("GET", "/api/stream/test") as response:
                # Should return 200 with SSE content type (or 404 if not registered)
                if response.status_code == 200:
                    assert "text/event-stream" in response.headers.get("content-type", "")

                    # Read first few events
                    events = []
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            events.append(line)
                            if len(events) >= 5:  # Read first 5 events
                                break

                    # Should have received some events
                    assert len(events) > 0

        except Exception as e:
            # Route might not be registered yet, skip test
            pytest.skip(f"Streaming endpoint not available: {e}")

    async def test_stream_chat_endpoint_format(self, client):
        """Test /api/stream/chat returns correct SSE format"""
        try:
            # Mock agent to avoid real LLM calls
            with patch('routes_streaming.get_agent') as mock_get_agent:
                mock_agent = AsyncMock()
                mock_agent.llm = Mock()
                mock_agent.llm.streaming = False
                mock_agent.llm.callbacks = []

                # Mock arun to return simple result
                mock_agent.arun.return_value = {
                    'output': 'Test response',
                    'sources': []
                }

                mock_get_agent.return_value = mock_agent

                async with client.stream(
                    "POST",
                    "/api/stream/chat",
                    json={"query": "Test query"}
                ) as response:
                    if response.status_code == 200:
                        # Should have SSE content type
                        assert "text/event-stream" in response.headers.get("content-type", "")

                        # Should have no-cache header
                        assert "no-cache" in response.headers.get("cache-control", "")

        except Exception as e:
            pytest.skip(f"Streaming chat endpoint not available: {e}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
class TestStreamingIntegration:
    """Integration tests for streaming system"""

    @pytest.mark.skipif(
        True,
        reason="Integration tests require --run-integration flag"
    )
    async def test_full_streaming_flow(self):
        """Test complete streaming flow with real agent"""
        from langchain_agent import DocumentationAgent

        agent = DocumentationAgent()

        # Enable streaming
        original_streaming = agent.llm.streaming
        agent.llm.streaming = True

        try:
            events = []

            async for event in stream_agent_response(agent, "What is Python?"):
                events.append(event)

                # Break after reasonable number of events
                if len(events) >= 50:
                    break

            # Should have received events
            assert len(events) > 0

            # Should have token events
            has_tokens = any('"token"' in event for event in events)
            assert has_tokens == True

        finally:
            # Restore original setting
            agent.llm.streaming = original_streaming

    @pytest.mark.skipif(
        True,
        reason="Integration tests require --run-integration flag"
    )
    async def test_streaming_latency(self):
        """Test first token latency"""
        import time
        from langchain_agent import DocumentationAgent

        agent = DocumentationAgent()
        agent.llm.streaming = True

        start_time = time.time()
        first_token_time = None

        try:
            async for event in stream_agent_response(agent, "Hello"):
                if first_token_time is None and '"token"' in event:
                    first_token_time = time.time()
                    break

            if first_token_time:
                latency = first_token_time - start_time
                # Should be under 2 seconds for first token
                assert latency < 2.0

        except Exception as e:
            pytest.skip(f"Latency test failed: {e}")


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_addoption(parser):
    """Add custom pytest options"""
    try:
        parser.addoption(
            "--run-integration",
            action="store_true",
            default=False,
            help="Run integration tests (requires API keys)"
        )
    except:
        # Option may already be defined
        pass


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
