"""
Performance Testing Script for RAG System

Tests response times and identifies bottlenecks.
Runs multiple queries and generates performance report.
"""

import asyncio
import time
import httpx
from statistics import mean, median
from typing import List, Dict, Any


# Test queries covering different scenarios
TEST_QUERIES = [
    "How to authenticate with GitHub API?",
    "What is the Stripe payment API?",
    "How to use OpenAI chat completions?",
    "What are React hooks?",
    "How to deploy Next.js?",
    "GitHub repository creation API",
    "FastAPI async endpoints",
    "Stripe webhook handling",
    "OpenAI embeddings API",
    "React context API",
    "How to handle errors in FastAPI?",
    "What is GitHub Actions?",
    "Stripe subscription management",
    "Next.js API routes",
    "React useEffect hook"
]


async def test_response_times(n_queries: int = 10, base_url: str = "http://127.0.0.1:8000"):
    """
    Test response times for multiple queries.

    Args:
        n_queries: Number of queries to test (default: 10)
        base_url: Base URL of the API (default: http://127.0.0.1:8000)

    Returns:
        List of response times in seconds
    """

    print("="*60)
    print("RAG SYSTEM PERFORMANCE TEST")
    print("="*60)
    print(f"Testing {n_queries} queries...")
    print(f"Base URL: {base_url}")
    print("="*60)

    times = []
    perf_breakdowns = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, query in enumerate(TEST_QUERIES[:n_queries]):
            print(f"\n[{i+1}/{n_queries}] Testing: {query[:50]}...")

            start = time.time()
            try:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={"query": query},
                    headers={"Cookie": f"session_id=perf_test_{i}"}
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    data = response.json()
                    server_time = data.get('response_time', 0)
                    cached = data.get('cached', False)

                    times.append(elapsed)

                    # Extract performance metrics if available
                    perf_metrics = data.get('perf_metrics', {})
                    if perf_metrics:
                        perf_breakdowns.append(perf_metrics)

                    print(f"  [OK] Total: {elapsed:.2f}s | Server: {server_time:.2f}s | Cached: {cached}")

                    # Show performance breakdown if available
                    if perf_metrics:
                        print(f"    - Cache Check: {perf_metrics.get('cache_check', 0):.3f}s")
                        print(f"    - Embedding: {perf_metrics.get('embedding_generation', 0):.3f}s")
                        print(f"    - ChromaDB: {perf_metrics.get('chromadb_query', 0):.3f}s")
                        print(f"    - LLM: {perf_metrics.get('llm_generation', 0):.3f}s")
                else:
                    print(f"  [ERROR] Status: {response.status_code}")

            except Exception as e:
                print(f"  [ERROR] Exception: {str(e)}")

    # Calculate statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    if not times:
        print("No successful queries to analyze!")
        return []

    print(f"Queries tested: {len(times)}")
    print(f"Average time: {mean(times):.2f}s")
    print(f"Median time: {median(times):.2f}s")
    print(f"Min time: {min(times):.2f}s")
    print(f"Max time: {max(times):.2f}s")

    if len(times) > 1:
        p95_index = int(len(times) * 0.95)
        print(f"P95 time: {sorted(times)[p95_index]:.2f}s")

    print("="*60)

    # Detailed breakdown analysis
    if perf_breakdowns:
        print("\nDETAILED PERFORMANCE BREAKDOWN (averages):")
        print("="*60)

        avg_cache = mean([p.get('cache_check', 0) for p in perf_breakdowns])
        avg_embedding = mean([p.get('embedding_generation', 0) for p in perf_breakdowns])
        avg_chromadb = mean([p.get('chromadb_query', 0) for p in perf_breakdowns])
        avg_llm = mean([p.get('llm_generation', 0) for p in perf_breakdowns])
        avg_context = mean([p.get('context_building', 0) for p in perf_breakdowns])
        avg_post = mean([p.get('post_processing', 0) for p in perf_breakdowns])

        print(f"Cache Check:        {avg_cache:.3f}s ({avg_cache/mean(times)*100:.1f}%)")
        print(f"Embedding Gen:      {avg_embedding:.3f}s ({avg_embedding/mean(times)*100:.1f}%)")
        print(f"ChromaDB Query:     {avg_chromadb:.3f}s ({avg_chromadb/mean(times)*100:.1f}%)")
        print(f"Context Building:   {avg_context:.3f}s ({avg_context/mean(times)*100:.1f}%)")
        print(f"LLM Generation:     {avg_llm:.3f}s ({avg_llm/mean(times)*100:.1f}%)")
        print(f"Post-processing:    {avg_post:.3f}s ({avg_post/mean(times)*100:.1f}%)")
        print("="*60)

        # Identify bottleneck
        bottlenecks = {
            "LLM Generation": avg_llm,
            "Embedding Generation": avg_embedding,
            "ChromaDB Query": avg_chromadb,
            "Context Building": avg_context
        }
        bottleneck = max(bottlenecks, key=bottlenecks.get)
        print(f"\nPrimary Bottleneck: {bottleneck} ({bottlenecks[bottleneck]:.3f}s)")

    # Success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)

    avg = mean(times)
    if avg <= 3.0:
        print("EXCELLENT - Target met (<=3s avg)")
        print("    System is performing optimally!")
    elif avg <= 5.0:
        print("GOOD - Within acceptable range (<=5s avg)")
        print("   System is performing well.")
    elif avg <= 8.0:
        print("ACCEPTABLE - Could be better (<=8s avg)")
        print("   Consider further optimization.")
    else:
        print("NEEDS IMPROVEMENT - Too slow (>8s avg)")
        print("   Significant optimization required.")

    print("="*60)

    return times


async def run_quick_test():
    """Run a quick 5-query test"""
    print("\n>>> Running QUICK TEST (5 queries)...\n")
    await test_response_times(n_queries=5)


async def run_full_test():
    """Run a comprehensive 15-query test"""
    print("\n>>> Running FULL TEST (15 queries)...\n")
    await test_response_times(n_queries=15)


async def run_cache_test():
    """Test cache performance with repeated queries"""
    print("\n>>> Running CACHE TEST (same query 3 times)...\n")
    print("="*60)
    print("Testing cache performance...")
    print("="*60)

    query = "How to authenticate with GitHub API?"

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(3):
            print(f"\n[{i+1}/3] Query: {query}")
            start = time.time()

            response = await client.post(
                "http://127.0.0.1:8000/api/chat",
                json={"query": query},
                headers={"Cookie": "session_id=cache_test"}
            )

            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                cached = data.get('cached', False)
                server_time = data.get('response_time', 0)

                cache_status = "HIT [FAST]" if cached else "MISS"
                print(f"  Cache: {cache_status}")
                print(f"  Total: {elapsed:.3f}s")
                print(f"  Server: {server_time:.3f}s")

                if cached and server_time > 0.1:
                    print("  [WARNING] Cache hit but still slow!")
                elif cached and server_time < 0.01:
                    print("  [OK] Excellent cache performance!")


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "quick":
            asyncio.run(run_quick_test())
        elif test_type == "full":
            asyncio.run(run_full_test())
        elif test_type == "cache":
            asyncio.run(run_cache_test())
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python performance_test.py [quick|full|cache]")
            sys.exit(1)
    else:
        # Default: run quick test
        asyncio.run(run_quick_test())


if __name__ == "__main__":
    main()
