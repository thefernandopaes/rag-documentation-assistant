"""
RAG Benchmark Suite - Comprehensive Performance and Quality Testing

Runs complete benchmark of RAG system including:
- Performance metrics (latency, throughput)
- Quality metrics (RAGAs scores)
- Retrieval metrics (precision, recall)
"""

import asyncio
import time
import json
import logging
from typing import Dict, List
from datetime import datetime
import numpy as np

from evaluation.ragas_evaluator import RAGEvaluator
from evaluation.metrics import CustomMetrics, LatencyMetrics, TokenMetrics, QualityMetrics

logger = logging.getLogger(__name__)


class RAGBenchmark:
    """
    Comprehensive benchmark suite for RAG system.

    Tests performance, quality, and retrieval accuracy.
    """

    def __init__(self, rag_engine):
        """
        Initialize benchmark.

        Args:
            rag_engine: AsyncRAGEngine instance to benchmark
        """
        self.rag_engine = rag_engine
        self.evaluator = RAGEvaluator()

    async def run_benchmark(self, save_results: bool = True) -> Dict:
        """
        Run complete benchmark suite.

        Args:
            save_results: Whether to save results to file

        Returns:
            Complete benchmark results
        """
        print("=" * 70)
        print("RAG SYSTEM BENCHMARK")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        results = {
            'timestamp': datetime.now().isoformat(),
            'performance': await self._benchmark_performance(),
            'quality': await self._benchmark_quality(),
            'retrieval': await self._benchmark_retrieval()
        }

        # Calculate overall scores
        results['summary'] = self._generate_summary(results)

        # Print summary
        self._print_summary(results)

        # Save results
        if save_results:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to: {filename}")

        return results

    async def _benchmark_performance(self) -> Dict:
        """
        Benchmark query performance.

        Tests:
        - Query latency (avg, p50, p95, p99)
        - Throughput
        - Token usage

        Returns:
            Performance metrics
        """
        print("\n1. PERFORMANCE BENCHMARK")
        print("-" * 70)

        test_queries = [
            "What is FastAPI?",
            "How to create an API endpoint?",
            "What is async/await?",
            "How to handle CORS?",
            "What is Pydantic?",
            "How to add authentication?",
            "What is dependency injection?",
            "How to handle file uploads?",
            "What is SQLAlchemy?",
            "How to use path parameters?"
        ]

        latencies = []
        token_counts = []

        print(f"Running {len(test_queries)} queries...")

        for i, query in enumerate(test_queries, 1):
            start = time.time()

            try:
                result = await self.rag_engine.generate_response(query)
                elapsed = time.time() - start

                latencies.append(elapsed)

                # Count tokens (approximate)
                tokens = len(result.get('answer', '').split()) * 1.3  # Rough estimate
                token_counts.append(int(tokens))

                print(f"  [{i}/{len(test_queries)}] {query[:40]}... → {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"Query failed: {e}")
                print(f"  [{i}/{len(test_queries)}] {query[:40]}... → ERROR")

        # Calculate statistics
        latency_stats = LatencyMetrics.calculate_statistics(latencies)

        performance = {
            'queries_tested': len(test_queries),
            'successful_queries': len(latencies),
            'latency': latency_stats,
            'token_usage': {
                'avg_tokens': int(np.mean(token_counts)) if token_counts else 0,
                'total_tokens': int(np.sum(token_counts)) if token_counts else 0
            },
            'throughput': {
                'queries_per_minute': round(60 / latency_stats['avg'], 1) if latency_stats['avg'] > 0 else 0
            }
        }

        print(f"\n  ✓ Average latency: {latency_stats['avg']:.2f}s")
        print(f"  ✓ P95 latency: {latency_stats['p95']:.2f}s")
        print(f"  ✓ Throughput: {performance['throughput']['queries_per_minute']:.1f} queries/min")

        return performance

    async def _benchmark_quality(self) -> Dict:
        """
        Benchmark answer quality using RAGAs.

        Tests:
        - Faithfulness
        - Answer relevancy
        - Context recall
        - Context precision
        - Answer similarity

        Returns:
            Quality metrics
        """
        print("\n2. QUALITY BENCHMARK (RAGAs)")
        print("-" * 70)

        print("Running RAGAs evaluation on test dataset...")

        try:
            results = await self.evaluator.run_evaluation(self.rag_engine)

            if 'error' in results:
                print(f"  ⚠️  RAGAs evaluation failed: {results['error']}")
                return results

            metrics = results.get('metrics', {})

            print(f"\n  Faithfulness:      {metrics.get('faithfulness', 0):.3f}")
            print(f"  Answer Relevancy:  {metrics.get('answer_relevancy', 0):.3f}")
            print(f"  Context Recall:    {metrics.get('context_recall', 0):.3f}")
            print(f"  Context Precision: {metrics.get('context_precision', 0):.3f}")
            print(f"  Answer Similarity: {metrics.get('answer_similarity', 0):.3f}")
            print(f"\n  Overall Score:     {results.get('overall_score', 0):.3f}")

            return results

        except Exception as e:
            logger.error(f"Quality benchmark error: {e}", exc_info=True)
            return {
                'error': str(e),
                'message': 'Quality evaluation failed'
            }

    async def _benchmark_retrieval(self) -> Dict:
        """
        Benchmark document retrieval quality.

        Tests:
        - Retrieval precision
        - Search relevance
        - Context quality

        Returns:
            Retrieval metrics
        """
        print("\n3. RETRIEVAL BENCHMARK")
        print("-" * 70)

        # Test retrieval with known queries and expected terms
        test_cases = [
            {
                'query': 'FastAPI framework',
                'expected_terms': ['fastapi', 'framework', 'api', 'python'],
                'min_terms': 2
            },
            {
                'query': 'async await Python',
                'expected_terms': ['async', 'await', 'asyncio', 'coroutine'],
                'min_terms': 2
            },
            {
                'query': 'CORS middleware',
                'expected_terms': ['cors', 'middleware', 'origin', 'header'],
                'min_terms': 2
            },
            {
                'query': 'Pydantic validation',
                'expected_terms': ['pydantic', 'validation', 'model', 'field'],
                'min_terms': 2
            },
            {
                'query': 'SQLAlchemy ORM',
                'expected_terms': ['sqlalchemy', 'orm', 'database', 'query'],
                'min_terms': 2
            }
        ]

        precisions = []
        recalls = []

        print(f"Testing retrieval with {len(test_cases)} queries...")

        for i, case in enumerate(test_cases, 1):
            try:
                # Search documents
                results = await self.rag_engine.search_documents(
                    case['query'],
                    n_results=5
                )

                if not results or 'documents' not in results:
                    print(f"  [{i}/{len(test_cases)}] {case['query'][:40]}... → NO RESULTS")
                    continue

                # Combine all retrieved text
                retrieved_text = " ".join(results['documents'][0]).lower()

                # Check how many expected terms appear
                matches = sum(
                    1 for term in case['expected_terms']
                    if term.lower() in retrieved_text
                )

                precision = matches / len(case['expected_terms'])
                recall = matches / case['min_terms'] if case['min_terms'] > 0 else 0

                precisions.append(precision)
                recalls.append(recall)

                status = "✓" if matches >= case['min_terms'] else "✗"
                print(f"  [{i}/{len(test_cases)}] {case['query'][:40]}... → {status} Precision: {precision:.2f}")

            except Exception as e:
                logger.error(f"Retrieval test error: {e}")
                print(f"  [{i}/{len(test_cases)}] {case['query'][:40]}... → ERROR")

        retrieval = {
            'queries_tested': len(test_cases),
            'avg_precision': round(np.mean(precisions), 3) if precisions else 0.0,
            'avg_recall': round(np.mean(recalls), 3) if recalls else 0.0,
            'f1_score': CustomMetrics.calculate_f1_score(
                np.mean(precisions) if precisions else 0.0,
                np.mean(recalls) if recalls else 0.0
            )
        }

        print(f"\n  ✓ Average Precision: {retrieval['avg_precision']:.3f}")
        print(f"  ✓ Average Recall:    {retrieval['avg_recall']:.3f}")
        print(f"  ✓ F1 Score:          {retrieval['f1_score']:.3f}")

        return retrieval

    def _generate_summary(self, results: Dict) -> Dict:
        """
        Generate overall summary from benchmark results.

        Args:
            results: Complete benchmark results

        Returns:
            Summary dictionary
        """
        performance = results.get('performance', {})
        quality = results.get('quality', {})
        retrieval = results.get('retrieval', {})

        # Calculate scores
        performance_score = self._calculate_performance_score(performance)
        quality_score = quality.get('overall_score', 0.0)
        retrieval_score = retrieval.get('f1_score', 0.0)

        # Overall score (weighted average)
        overall = (performance_score * 0.3 + quality_score * 0.5 + retrieval_score * 0.2)

        return {
            'performance_score': round(performance_score, 3),
            'quality_score': round(quality_score, 3),
            'retrieval_score': round(retrieval_score, 3),
            'overall_score': round(overall, 3),
            'grade': QualityMetrics.categorize_score(overall),
            'meets_targets': self._check_targets(results)
        }

    def _calculate_performance_score(self, performance: Dict) -> float:
        """
        Calculate performance score from metrics.

        Args:
            performance: Performance metrics

        Returns:
            Score (0-1)
        """
        latency = performance.get('latency', {})
        avg_latency = latency.get('avg', 10.0)

        # Target: < 2s average latency
        # Score: 1.0 at 1s, 0.5 at 2s, 0.0 at 4s+
        if avg_latency <= 1.0:
            return 1.0
        elif avg_latency <= 2.0:
            return 1.0 - ((avg_latency - 1.0) / 2.0)
        elif avg_latency <= 4.0:
            return 0.5 - ((avg_latency - 2.0) / 4.0)
        else:
            return 0.0

    def _check_targets(self, results: Dict) -> Dict:
        """
        Check if system meets target metrics.

        Args:
            results: Benchmark results

        Returns:
            Dictionary of target checks
        """
        performance = results.get('performance', {})
        quality = results.get('quality', {})
        retrieval = results.get('retrieval', {})

        latency = performance.get('latency', {})
        metrics = quality.get('metrics', {})

        return {
            'avg_latency_under_2s': latency.get('avg', 10) < 2.0,
            'p95_latency_under_3s': latency.get('p95', 10) < 3.0,
            'faithfulness_over_0_85': metrics.get('faithfulness', 0) > 0.85,
            'answer_relevancy_over_0_80': metrics.get('answer_relevancy', 0) > 0.80,
            'retrieval_precision_over_0_80': retrieval.get('avg_precision', 0) > 0.80
        }

    def _print_summary(self, results: Dict):
        """
        Print benchmark summary.

        Args:
            results: Complete benchmark results
        """
        summary = results['summary']

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\nPerformance Score: {summary['performance_score']:.3f}")
        print(f"Quality Score:     {summary['quality_score']:.3f}")
        print(f"Retrieval Score:   {summary['retrieval_score']:.3f}")
        print(f"\nOverall Score:     {summary['overall_score']:.3f} ({summary['grade']})")

        print("\nTarget Metrics:")
        targets = summary['meets_targets']
        for target, met in targets.items():
            status = "✓" if met else "✗"
            print(f"  {status} {target.replace('_', ' ').title()}")

        print("\n" + "=" * 70)


# ============================================================================
# CLI USAGE
# ============================================================================

async def main():
    """Run benchmark from command line"""
    print("Initializing RAG engine...")

    from rag_engine_async import AsyncRAGEngine

    engine = AsyncRAGEngine()
    benchmark = RAGBenchmark(engine)

    print("Starting benchmark...\n")
    await benchmark.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
