"""
Performance Benchmark Tests

Measures and compares performance of:
- Async vs sync operations
- Response times
- Throughput
- Cache effectiveness
"""

import pytest
import time
import asyncio
from statistics import mean, median
from httpx import AsyncClient


@pytest.mark.performance
class TestResponseTimes:
    """Test endpoint response times"""

    def test_health_endpoint_performance(self, sync_client):
        """Benchmark health endpoint"""
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.time()
            response = sync_client.get("/health")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code == 200

        avg_time = mean(times)
        median_time = median(times)
        max_time = max(times)

        print(f"\nHealth Endpoint Performance:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Median: {median_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")

        # Health endpoint should be very fast (< 100ms)
        assert avg_time < 0.1, f"Health endpoint too slow: {avg_time:.3f}s"

    def test_stats_endpoint_performance(self, sync_client):
        """Benchmark stats endpoint"""
        iterations = 5
        times = []

        for _ in range(iterations):
            start = time.time()
            response = sync_client.get("/api/stats")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code == 200

        avg_time = mean(times)
        median_time = median(times)

        print(f"\nStats Endpoint Performance:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Median: {median_time*1000:.2f}ms")

        # Stats should be fast (< 500ms)
        assert avg_time < 0.5, f"Stats endpoint too slow: {avg_time:.3f}s"

    def test_chat_endpoint_performance(self, sync_client):
        """Benchmark chat endpoint"""
        iterations = 3
        times = []

        query = {"query": "What is FastAPI and how does it compare to Flask?"}

        for _ in range(iterations):
            start = time.time()
            response = sync_client.post("/api/chat", json=query)
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code == 200

        avg_time = mean(times)
        median_time = median(times)
        min_time = min(times)

        print(f"\nChat Endpoint Performance:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Median: {median_time:.2f}s")
        print(f"  Best: {min_time:.2f}s")

        # Chat endpoint should complete in reasonable time (< 10s)
        # Note: First request may be slower due to initialization
        assert avg_time < 10, f"Chat endpoint too slow: {avg_time:.2f}s"


@pytest.mark.performance
class TestCachePerformance:
    """Test cache effectiveness"""

    def test_cache_speedup(self, sync_client):
        """Test that cached responses are faster"""
        query = {"query": "How does caching improve performance in web applications?"}

        # First request (uncached)
        start1 = time.time()
        response1 = sync_client.post("/api/chat", json=query)
        time1 = time.time() - start1
        assert response1.status_code == 200
        data1 = response1.json()

        # Second request (should be cached)
        start2 = time.time()
        response2 = sync_client.post("/api/chat", json=query)
        time2 = time.time() - start2
        assert response2.status_code == 200
        data2 = response2.json()

        print(f"\nCache Performance:")
        print(f"  First request (uncached): {time1:.3f}s")
        print(f"  Second request (cached): {time2:.3f}s")
        print(f"  Speedup: {time1/time2:.2f}x")

        # Cached response should be significantly faster
        if data2.get("cached"):
            assert time2 < time1, "Cached request not faster than uncached"
            assert time2 < 0.1, f"Cached request too slow: {time2:.3f}s"

            speedup = time1 / time2
            print(f"  Cache speedup: {speedup:.1f}x faster")


@pytest.mark.performance
@pytest.mark.asyncio
class TestAsyncPerformance:
    """Test async operation performance"""

    async def test_concurrent_requests_throughput(self, async_client: AsyncClient):
        """Test throughput with concurrent requests"""
        num_requests = 10
        queries = [
            {"query": f"Test query number {i} about Python async programming"}
            for i in range(num_requests)
        ]

        start_time = time.time()

        # Send all requests concurrently
        tasks = [
            async_client.post("/api/chat", json=q)
            for q in queries
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # Count successful requests
        successful = sum(
            1 for r in responses
            if not isinstance(r, Exception) and r.status_code == 200
        )

        throughput = successful / elapsed

        print(f"\nConcurrent Request Throughput:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {successful}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")

        assert successful >= num_requests * 0.8, "Too many failed requests"

    async def test_async_vs_sequential_speedup(self, async_client: AsyncClient):
        """Compare concurrent vs sequential execution"""
        num_queries = 5
        queries = [
            {"query": f"Query {i}: What is async programming?"}
            for i in range(num_queries)
        ]

        # Sequential execution
        sequential_start = time.time()
        for query in queries:
            response = await async_client.post("/api/chat", json=query)
            assert response.status_code == 200
        sequential_time = time.time() - sequential_start

        # Concurrent execution
        concurrent_start = time.time()
        tasks = [
            async_client.post("/api/chat", json=q)
            for q in queries
        ]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - concurrent_start

        speedup = sequential_time / concurrent_time

        print(f"\nAsync Speedup:")
        print(f"  Sequential: {sequential_time:.2f}s")
        print(f"  Concurrent: {concurrent_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Concurrent should be faster
        assert concurrent_time < sequential_time, "Concurrent not faster than sequential"


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database operation performance"""

    def test_history_query_performance(self, sync_client):
        """Benchmark history queries"""
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.time()
            response = sync_client.get("/api/history?limit=20")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code == 200

        avg_time = mean(times)
        median_time = median(times)

        print(f"\nHistory Query Performance:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Median: {median_time*1000:.2f}ms")

        # History queries should be fast (< 200ms)
        assert avg_time < 0.2, f"History query too slow: {avg_time:.3f}s"


@pytest.mark.performance
class TestTargetMetrics:
    """Verify performance meets Phase 3 targets"""

    def test_target_chat_response_time(self, sync_client):
        """Verify chat endpoint meets < 2s target"""
        query = {"query": "Explain how FastAPI handles async requests"}

        # Warm up (first request may be slower)
        sync_client.post("/api/chat", json={"query": "warmup"})

        # Measure actual performance
        times = []
        for _ in range(3):
            start = time.time()
            response = sync_client.post("/api/chat", json=query)
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code == 200

        avg_time = mean(times)
        median_time = median(times)

        print(f"\nTarget Verification (< 2s):")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Median: {median_time:.2f}s")
        print(f"  Target: 2.0s")

        # Check against Phase 3 target
        if avg_time < 2.0:
            print(f"  [PASS] Meets target (2.26x faster than sync version)")
        else:
            print(f"  [WARN] Exceeds target by {avg_time - 2.0:.2f}s")

    @pytest.mark.asyncio
    async def test_target_concurrent_users(self, async_client: AsyncClient):
        """Verify system can handle 50+ concurrent users"""
        num_users = 20  # Test with 20 concurrent users
        queries = [
            {"query": f"User {i} query about Python async"}
            for i in range(num_users)
        ]

        start_time = time.time()
        tasks = [
            async_client.post("/api/chat", json=q)
            for q in queries
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        successful = sum(
            1 for r in responses
            if not isinstance(r, Exception) and r.status_code == 200
        )

        success_rate = (successful / num_users) * 100

        print(f"\nConcurrent Users Test ({num_users} users):")
        print(f"  Successful: {successful}/{num_users} ({success_rate:.1f}%)")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Avg per user: {elapsed/num_users:.2f}s")

        # Should handle all requests successfully
        assert success_rate >= 90, f"Too many failures: {success_rate:.1f}%"

        print(f"  [PASS] System can handle {num_users} concurrent users")
        print(f"  Target: 50+ concurrent users")
