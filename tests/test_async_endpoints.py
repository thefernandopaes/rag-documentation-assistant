"""
Async Endpoint Integration Tests

Tests all FastAPI endpoints with async patterns:
- /health
- /api/chat
- /api/feedback
- /api/history
- /api/stats
- /api/initialize
"""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_sync(self, sync_client):
        """Test health endpoint (sync client)"""
        response = sync_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
        assert data["framework"] == "fastapi"
        assert data["async"] is True

    @pytest.mark.asyncio
    async def test_health_async(self, async_client: AsyncClient):
        """Test health endpoint (async client)"""
        response = await async_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data


class TestChatEndpoint:
    """Test /api/chat endpoint"""

    def test_chat_valid_query(self, sync_client, sample_chat_request):
        """Test chat with valid query"""
        response = sync_client.post("/api/chat", json=sample_chat_request)
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert "response_time" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert isinstance(data["sources"], list)
        assert isinstance(data["response_time"], (int, float))

    def test_chat_min_length_validation(self, sync_client):
        """Test minimum query length validation"""
        response = sync_client.post(
            "/api/chat",
            json={"query": "hi"}  # Too short (< 3 chars)
        )
        # Should return 422 (Pydantic validation error)
        assert response.status_code == 422

    def test_chat_max_length_validation(self, sync_client):
        """Test maximum query length validation"""
        long_query = "a" * 501  # Exceeds 500 char limit
        response = sync_client.post(
            "/api/chat",
            json={"query": long_query}
        )
        assert response.status_code == 422

    def test_chat_xss_detection(self, sync_client):
        """Test XSS attack detection"""
        xss_query = "<script>alert('xss')</script>"
        response = sync_client.post(
            "/api/chat",
            json={"query": xss_query}
        )
        # Should be blocked by security validation
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_chat_async(self, async_client: AsyncClient):
        """Test chat endpoint with async client"""
        response = await async_client.post(
            "/api/chat",
            json={"query": "What is FastAPI?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_chat_missing_query(self, sync_client):
        """Test chat without query field"""
        response = sync_client.post("/api/chat", json={})
        assert response.status_code == 422  # Validation error

    def test_chat_caching(self, sync_client):
        """Test response caching"""
        query = {"query": "How does caching work in FastAPI?"}

        # First request (uncached)
        response1 = sync_client.post("/api/chat", json=query)
        assert response1.status_code == 200
        data1 = response1.json()

        # Second request (should be cached)
        response2 = sync_client.post("/api/chat", json=query)
        assert response2.status_code == 200
        data2 = response2.json()

        # Cache should make second request faster
        if "cached" in data2:
            assert data2["cached"] is True
            assert data2["response_time"] < data1["response_time"]


class TestFeedbackEndpoint:
    """Test /api/feedback endpoint"""

    def test_feedback_valid(self, sync_client):
        """Test valid feedback submission"""
        # First create a conversation
        chat_response = sync_client.post(
            "/api/chat",
            json={"query": "Test query for feedback"}
        )
        assert chat_response.status_code == 200

        # Get conversation ID from history
        history_response = sync_client.get("/api/history")
        if history_response.status_code == 200:
            history = history_response.json()
            if history and len(history) > 0:
                conversation_id = history[0]["id"]

                # Submit feedback
                feedback_response = sync_client.post(
                    "/api/feedback",
                    json={
                        "conversation_id": conversation_id,
                        "feedback": 1
                    }
                )
                assert feedback_response.status_code == 200

    def test_feedback_invalid_conversation(self, sync_client):
        """Test feedback for non-existent conversation"""
        response = sync_client.post(
            "/api/feedback",
            json={
                "conversation_id": "nonexistent-id-12345",
                "feedback": 1
            }
        )
        # Should return 404 (not found)
        assert response.status_code == 404

    def test_feedback_invalid_value(self, sync_client):
        """Test feedback with invalid value"""
        response = sync_client.post(
            "/api/feedback",
            json={
                "conversation_id": "test-id",
                "feedback": 5  # Should be -1, 0, or 1
            }
        )
        # Should return 422 (validation error)
        assert response.status_code == 422


class TestHistoryEndpoint:
    """Test /api/history endpoint"""

    def test_history_empty(self, sync_client):
        """Test history with no conversations"""
        response = sync_client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_history_after_chat(self, sync_client):
        """Test history after creating conversation"""
        # Create conversation
        sync_client.post(
            "/api/chat",
            json={"query": "Test history query"}
        )

        # Get history
        response = sync_client.get("/api/history")
        assert response.status_code == 200
        history = response.json()
        assert isinstance(history, list)

        if history:
            first_entry = history[0]
            assert "id" in first_entry
            assert "query" in first_entry
            assert "response" in first_entry
            assert "created_at" in first_entry

    def test_history_limit(self, sync_client):
        """Test history with limit parameter"""
        response = sync_client.get("/api/history?limit=5")
        assert response.status_code == 200
        history = response.json()
        assert len(history) <= 5

    @pytest.mark.asyncio
    async def test_history_async(self, async_client: AsyncClient):
        """Test history endpoint with async client"""
        response = await async_client.get("/api/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestStatsEndpoint:
    """Test /api/stats endpoint"""

    def test_stats_structure(self, sync_client):
        """Test stats endpoint structure"""
        response = sync_client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert "documents" in data
        assert "conversations" in data
        assert "cache" in data
        assert "system" in data

        # Check system info
        assert data["system"]["framework"] == "fastapi"
        assert data["system"]["async"] is True
        assert data["system"]["version"] == "2.0.0"

    def test_stats_documents(self, sync_client):
        """Test stats documents section"""
        response = sync_client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        docs = data["documents"]
        assert "document_count" in docs
        assert isinstance(docs["document_count"], int)
        assert docs["document_count"] >= 0

    def test_stats_conversations(self, sync_client):
        """Test stats conversations section"""
        response = sync_client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        convs = data["conversations"]
        assert "total" in convs
        assert "avg_response_time" in convs
        assert isinstance(convs["total"], int)

    @pytest.mark.asyncio
    async def test_stats_async(self, async_client: AsyncClient):
        """Test stats endpoint with async client"""
        response = await async_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data


@pytest.mark.integration
class TestEndpointIntegration:
    """Integration tests across multiple endpoints"""

    def test_full_workflow(self, sync_client):
        """Test complete user workflow"""
        # 1. Check health
        health_response = sync_client.get("/health")
        assert health_response.status_code == 200

        # 2. Check stats
        stats_response = sync_client.get("/api/stats")
        assert stats_response.status_code == 200

        # 3. Send chat query
        chat_response = sync_client.post(
            "/api/chat",
            json={"query": "What is async/await in Python?"}
        )
        assert chat_response.status_code == 200

        # 4. Get history
        history_response = sync_client.get("/api/history")
        assert history_response.status_code == 200

        # 5. Submit feedback (if conversation exists)
        history = history_response.json()
        if history and len(history) > 0:
            conversation_id = history[0]["id"]
            feedback_response = sync_client.post(
                "/api/feedback",
                json={
                    "conversation_id": conversation_id,
                    "feedback": 1
                }
            )
            assert feedback_response.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self, async_client: AsyncClient):
        """Test multiple concurrent chat requests"""
        import asyncio

        queries = [
            {"query": "What is FastAPI?"},
            {"query": "What is Docker?"},
            {"query": "What is React?"},
        ]

        # Send all requests concurrently
        tasks = [
            async_client.post("/api/chat", json=q)
            for q in queries
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
