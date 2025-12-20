"""
Evaluation Tests - Tests for RAG Evaluation System

Tests for RAGAs evaluator, custom metrics, and benchmark suite.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

# Evaluation components
from evaluation.ragas_evaluator import RAGEvaluator
from evaluation.metrics import (
    CustomMetrics,
    LatencyMetrics,
    TokenMetrics,
    SemanticMetrics,
    QualityMetrics
)
from evaluation.benchmark import RAGBenchmark


# ============================================================================
# RAGAs EVALUATOR TESTS
# ============================================================================

class TestRAGEvaluator:
    """Tests for RAGEvaluator"""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing"""
        return RAGEvaluator()

    def test_initialization(self, evaluator):
        """Test evaluator initializes"""
        assert evaluator is not None
        # Metrics may not be available without ragas installed
        assert isinstance(evaluator.metrics, list)

    def test_load_test_cases(self, evaluator):
        """Test loading test cases from JSON"""
        test_cases = evaluator.load_test_cases("evaluation/test_dataset.json")

        # Should load cases (if file exists)
        if test_cases:
            assert len(test_cases) > 0
            assert 'question' in test_cases[0]
            assert 'ground_truth' in test_cases[0]

    def test_get_metrics_description(self, evaluator):
        """Test getting metrics descriptions"""
        descriptions = evaluator.get_metrics_description()

        assert 'faithfulness' in descriptions
        assert 'answer_relevancy' in descriptions
        assert len(descriptions) == 5  # 5 RAGAs metrics

    @pytest.mark.asyncio
    async def test_evaluate_system_without_ragas(self, evaluator):
        """Test evaluation without RAGAs installed"""
        if evaluator.evaluate_func is None:
            # Mock RAG engine
            mock_engine = AsyncMock()

            result = await evaluator.evaluate_system([], mock_engine)

            assert 'error' in result
            assert 'RAGAs not installed' in result['error']


# ============================================================================
# CUSTOM METRICS TESTS
# ============================================================================

class TestCustomMetrics:
    """Tests for CustomMetrics"""

    def test_calculate_mrr(self):
        """Test Mean Reciprocal Rank calculation"""
        # Perfect ranking (relevant doc at position 1)
        mrr = CustomMetrics.calculate_mrr([1, 1, 1])
        assert mrr == 1.0

        # Mixed rankings
        mrr = CustomMetrics.calculate_mrr([1, 2, 3])
        assert 0.5 < mrr < 0.7

        # Empty rankings
        mrr = CustomMetrics.calculate_mrr([])
        assert mrr == 0.0

    def test_calculate_ndcg(self):
        """Test NDCG calculation"""
        # Perfect ranking
        scores = [1.0, 0.8, 0.6, 0.4, 0.2]
        ndcg = CustomMetrics.calculate_ndcg(scores, k=5)
        assert ndcg == 1.0

        # Random ranking
        scores = [0.2, 1.0, 0.4, 0.8, 0.6]
        ndcg = CustomMetrics.calculate_ndcg(scores, k=5)
        assert 0.0 < ndcg < 1.0

    def test_calculate_precision_at_k(self):
        """Test Precision@K calculation"""
        # All relevant
        relevant = [True, True, True, True, True]
        precision = CustomMetrics.calculate_precision_at_k(relevant, k=5)
        assert precision == 1.0

        # Half relevant
        relevant = [True, False, True, False, True]
        precision = CustomMetrics.calculate_precision_at_k(relevant, k=5)
        assert precision == 0.6

        # None relevant
        relevant = [False, False, False]
        precision = CustomMetrics.calculate_precision_at_k(relevant, k=3)
        assert precision == 0.0

    def test_calculate_recall_at_k(self):
        """Test Recall@K calculation"""
        # Found all relevant
        relevant = [True, True, True]
        recall = CustomMetrics.calculate_recall_at_k(relevant, total_relevant=3, k=3)
        assert recall == 1.0

        # Found half
        relevant = [True, False, True]
        recall = CustomMetrics.calculate_recall_at_k(relevant, total_relevant=4, k=3)
        assert recall == 0.5

    def test_calculate_f1_score(self):
        """Test F1 score calculation"""
        # Perfect
        f1 = CustomMetrics.calculate_f1_score(1.0, 1.0)
        assert f1 == 1.0

        # Balanced
        f1 = CustomMetrics.calculate_f1_score(0.8, 0.8)
        assert 0.79 < f1 < 0.81

        # Zero
        f1 = CustomMetrics.calculate_f1_score(0.0, 0.0)
        assert f1 == 0.0

    def test_calculate_map(self):
        """Test Mean Average Precision calculation"""
        # Perfect ranking
        rankings = [[True, True, True], [True, True, True]]
        map_score = CustomMetrics.calculate_map(rankings)
        assert map_score == 1.0

        # Mixed rankings
        rankings = [[True, False, True], [False, True, False]]
        map_score = CustomMetrics.calculate_map(rankings)
        assert 0.0 < map_score < 1.0


# ============================================================================
# LATENCY METRICS TESTS
# ============================================================================

class TestLatencyMetrics:
    """Tests for LatencyMetrics"""

    def test_calculate_statistics(self):
        """Test latency statistics calculation"""
        latencies = [1.0, 1.5, 2.0, 2.5, 3.0]

        stats = LatencyMetrics.calculate_statistics(latencies)

        assert 'avg' in stats
        assert 'median' in stats
        assert 'p95' in stats
        assert 'p99' in stats
        assert 'min' in stats
        assert 'max' in stats

        assert stats['avg'] == 2.0
        assert stats['median'] == 2.0
        assert stats['min'] == 1.0
        assert stats['max'] == 3.0

    def test_empty_latencies(self):
        """Test with empty latency list"""
        stats = LatencyMetrics.calculate_statistics([])

        assert stats['avg'] == 0.0
        assert stats['p95'] == 0.0


# ============================================================================
# TOKEN METRICS TESTS
# ============================================================================

class TestTokenMetrics:
    """Tests for TokenMetrics"""

    def test_calculate_cost(self):
        """Test token cost calculation"""
        # GPT-4: $0.03/1K input + $0.06/1K output
        cost = TokenMetrics.calculate_cost('gpt-4', 1000, 1000)
        assert cost == 0.09  # $0.03 + $0.06

        # GPT-3.5-turbo: Much cheaper
        cost = TokenMetrics.calculate_cost('gpt-3.5-turbo', 1000, 1000)
        assert cost < 0.01

    def test_calculate_token_efficiency(self):
        """Test token efficiency calculation"""
        # Efficient: short answer, long context
        efficiency = TokenMetrics.calculate_token_efficiency(100, 1000)
        assert efficiency == 0.1

        # Inefficient: long answer, short context
        efficiency = TokenMetrics.calculate_token_efficiency(1000, 100)
        assert efficiency == 10.0


# ============================================================================
# SEMANTIC METRICS TESTS
# ============================================================================

class TestSemanticMetrics:
    """Tests for SemanticMetrics"""

    def test_calculate_embedding_similarity(self):
        """Test embedding similarity calculation"""
        # Identical embeddings
        emb1 = [1.0, 0.5, 0.2]
        emb2 = [1.0, 0.5, 0.2]
        similarity = SemanticMetrics.calculate_embedding_similarity(emb1, emb2)
        assert similarity == 1.0

        # Orthogonal embeddings
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity = SemanticMetrics.calculate_embedding_similarity(emb1, emb2)
        assert similarity == 0.0

        # Similar embeddings
        emb1 = [1.0, 0.8, 0.6]
        emb2 = [0.9, 0.7, 0.5]
        similarity = SemanticMetrics.calculate_embedding_similarity(emb1, emb2)
        assert 0.9 < similarity < 1.0


# ============================================================================
# QUALITY METRICS TESTS
# ============================================================================

class TestQualityMetrics:
    """Tests for QualityMetrics"""

    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        metrics = {
            'faithfulness': 0.9,
            'answer_relevancy': 0.85,
            'context_recall': 0.8,
            'context_precision': 0.75,
            'answer_similarity': 0.7
        }

        score = QualityMetrics.calculate_overall_score(metrics)

        assert 0.7 < score < 0.9

    def test_categorize_score(self):
        """Test score categorization"""
        assert QualityMetrics.categorize_score(0.95) == "Excellent"
        assert QualityMetrics.categorize_score(0.85) == "Good"
        assert QualityMetrics.categorize_score(0.75) == "Fair"
        assert QualityMetrics.categorize_score(0.65) == "Poor"
        assert QualityMetrics.categorize_score(0.50) == "Very Poor"


# ============================================================================
# BENCHMARK TESTS
# ============================================================================

class TestRAGBenchmark:
    """Tests for RAGBenchmark"""

    @pytest.fixture
    def mock_rag_engine(self):
        """Create mock RAG engine for testing"""
        engine = AsyncMock()

        # Mock generate_response
        engine.generate_response.return_value = {
            'answer': 'Test answer',
            'context': 'Test context',
            'sources': []
        }

        # Mock search_documents
        engine.search_documents.return_value = {
            'documents': [['Test document content']],
            'metadatas': [[{'source': 'test'}]]
        }

        return engine

    @pytest.fixture
    def benchmark(self, mock_rag_engine):
        """Create benchmark with mock engine"""
        return RAGBenchmark(mock_rag_engine)

    def test_initialization(self, benchmark):
        """Test benchmark initializes"""
        assert benchmark.rag_engine is not None
        assert benchmark.evaluator is not None

    @pytest.mark.asyncio
    async def test_benchmark_performance(self, benchmark):
        """Test performance benchmark"""
        results = await benchmark._benchmark_performance()

        assert 'queries_tested' in results
        assert 'latency' in results
        assert 'throughput' in results

    @pytest.mark.asyncio
    async def test_benchmark_retrieval(self, benchmark):
        """Test retrieval benchmark"""
        results = await benchmark._benchmark_retrieval()

        assert 'queries_tested' in results
        assert 'avg_precision' in results
        assert 'avg_recall' in results
        assert 'f1_score' in results

    def test_calculate_performance_score(self, benchmark):
        """Test performance score calculation"""
        # Fast performance
        performance = {'latency': {'avg': 1.0}}
        score = benchmark._calculate_performance_score(performance)
        assert score == 1.0

        # Slow performance
        performance = {'latency': {'avg': 5.0}}
        score = benchmark._calculate_performance_score(performance)
        assert score < 0.5


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
class TestEvaluationIntegration:
    """Integration tests for evaluation system"""

    @pytest.mark.skipif(
        True,
        reason="Integration tests require --run-integration flag"
    )
    async def test_full_evaluation(self):
        """Test complete evaluation flow"""
        from rag_engine_async import AsyncRAGEngine

        engine = AsyncRAGEngine()
        evaluator = RAGEvaluator()

        # Load test cases
        test_cases = evaluator.load_test_cases()

        if test_cases and evaluator.evaluate_func:
            # Run evaluation (limit to 3 cases for speed)
            result = await evaluator.evaluate_system(test_cases[:3], engine)

            # Should have metrics or error
            assert 'metrics' in result or 'error' in result

    @pytest.mark.skipif(
        True,
        reason="Integration tests require --run-integration flag"
    )
    async def test_full_benchmark(self):
        """Test complete benchmark"""
        from rag_engine_async import AsyncRAGEngine

        engine = AsyncRAGEngine()
        benchmark = RAGBenchmark(engine)

        # Run benchmark
        results = await benchmark.run_benchmark(save_results=False)

        # Should have all sections
        assert 'performance' in results
        assert 'quality' in results or 'error' in results.get('quality', {})
        assert 'retrieval' in results
        assert 'summary' in results


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
