"""
Observability Tests - Tests for LangSmith Integration and Metrics

Tests for tracing, callbacks, and metrics collection.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# Observability components
from langsmith_config import LangSmithConfig, setup_langsmith
from callbacks.tracing import MetricsCallbackHandler, DebugCallbackHandler
from monitoring.metrics import MetricsCollector, RealTimeMetrics, get_realtime_metrics


# ============================================================================
# LANGSMITH CONFIG TESTS
# ============================================================================

class TestLangSmithConfig:
    """Tests for LangSmith configuration"""

    def test_is_enabled_false_by_default(self):
        """Test tracing is disabled by default"""
        with patch.dict(os.environ, {'LANGSMITH_TRACING': 'false'}, clear=True):
            assert LangSmithConfig.is_enabled() == False

    def test_is_enabled_with_env_var(self):
        """Test tracing enabled with environment variable"""
        with patch.dict(os.environ, {'LANGSMITH_TRACING': 'true', 'LANGSMITH_API_KEY': 'test-key'}):
            assert LangSmithConfig.is_enabled() == True

    def test_is_enabled_requires_api_key(self):
        """Test tracing requires API key"""
        with patch.dict(os.environ, {'LANGSMITH_TRACING': 'true'}, clear=True):
            # Should be False without API key
            assert LangSmithConfig.is_enabled() == False

    def test_get_client_when_disabled(self):
        """Test get_client returns None when disabled"""
        with patch.dict(os.environ, {'LANGSMITH_TRACING': 'false'}, clear=True):
            client = LangSmithConfig.get_client()
            assert client is None

    def test_get_callbacks_when_disabled(self):
        """Test get_callbacks returns empty list when disabled"""
        with patch.dict(os.environ, {'LANGSMITH_TRACING': 'false'}, clear=True):
            callbacks = LangSmithConfig.get_callbacks()
            assert callbacks == []

    def test_get_environment_config(self):
        """Test environment configuration retrieval"""
        with patch.dict(os.environ, {
            'LANGSMITH_TRACING': 'true',
            'LANGSMITH_PROJECT': 'test-project',
            'LANGSMITH_API_KEY': 'test-key'
        }):
            config = LangSmithConfig.get_environment_config()

            assert config['enabled'] == True
            assert config['project'] == 'test-project'
            assert config['api_key_set'] == True


# ============================================================================
# METRICS CALLBACK TESTS
# ============================================================================

class TestMetricsCallbackHandler:
    """Tests for MetricsCallbackHandler"""

    @pytest.fixture
    def handler(self):
        """Create callback handler for testing"""
        return MetricsCallbackHandler()

    def test_initialization(self, handler):
        """Test handler initializes with zero metrics"""
        metrics = handler.get_metrics()

        assert metrics['calls']['llm'] == 0
        assert metrics['calls']['tool'] == 0
        assert metrics['tokens']['total'] == 0
        assert metrics['cost']['total'] == 0.0

    def test_llm_call_tracking(self, handler):
        """Test LLM call tracking"""
        # Simulate LLM start
        handler.on_llm_start(
            {'name': 'gpt-4'},
            ['Test prompt'],
            run_id='test-run-1'
        )

        # Create mock response
        mock_response = Mock()
        mock_response.llm_output = {
            'token_usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            },
            'model_name': 'gpt-4'
        }

        # Simulate LLM end
        handler.on_llm_end(mock_response, run_id='test-run-1')

        # Check metrics
        metrics = handler.get_metrics()

        assert metrics['calls']['llm'] == 1
        assert metrics['tokens']['total'] == 30
        assert metrics['tokens']['prompt'] == 10
        assert metrics['tokens']['completion'] == 20
        assert metrics['cost']['total'] > 0.0  # Should have some cost

    def test_tool_call_tracking(self, handler):
        """Test tool call tracking"""
        handler.on_tool_start({'name': 'test_tool'}, 'test input')
        handler.on_tool_end('test output')

        metrics = handler.get_metrics()

        assert metrics['calls']['tool'] == 1

    def test_error_tracking(self, handler):
        """Test error tracking"""
        error = Exception("Test error")

        handler.on_llm_error(error, run_id='test-run-error')

        metrics = handler.get_metrics()

        assert metrics['errors']['count'] == 1
        assert 'Test error' in metrics['errors']['details'][0]['error']

    def test_cost_calculation(self, handler):
        """Test cost calculation for different models"""
        # Test GPT-4 pricing
        cost = handler._calculate_cost('gpt-4', 1000, 1000)

        # GPT-4: $0.03/1K input + $0.06/1K output = $0.09 total
        assert cost == pytest.approx(0.09, rel=0.01)

        # Test GPT-3.5-turbo pricing
        cost = handler._calculate_cost('gpt-3.5-turbo', 1000, 1000)

        # GPT-3.5-turbo: $0.0005/1K input + $0.0015/1K output = $0.002 total
        assert cost == pytest.approx(0.002, rel=0.01)

    def test_reset(self, handler):
        """Test metrics reset"""
        # Add some metrics
        handler.on_llm_start({'name': 'gpt-4'}, ['test'], run_id='test')
        handler.on_tool_start({'name': 'tool'}, 'input')

        # Reset
        handler.reset()

        # Check all zero
        metrics = handler.get_metrics()

        assert metrics['calls']['llm'] == 0
        assert metrics['calls']['tool'] == 0
        assert metrics['tokens']['total'] == 0


# ============================================================================
# DEBUG CALLBACK TESTS
# ============================================================================

class TestDebugCallbackHandler:
    """Tests for DebugCallbackHandler"""

    @pytest.fixture
    def handler(self):
        """Create debug handler for testing"""
        return DebugCallbackHandler(verbose=True)

    def test_initialization(self, handler):
        """Test handler initializes"""
        assert handler.verbose == True

    def test_callbacks_execute_without_error(self, handler):
        """Test callbacks execute without errors"""
        # Should not raise exceptions
        handler.on_llm_start({'name': 'gpt-4'}, ['test prompt'])
        handler.on_llm_end(Mock())
        handler.on_tool_start({'name': 'tool'}, 'input')
        handler.on_tool_end('output')


# ============================================================================
# METRICS COLLECTOR TESTS
# ============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector"""

    @pytest.fixture
    def collector(self):
        """Create metrics collector for testing"""
        return MetricsCollector()

    @pytest.mark.asyncio
    async def test_daily_metrics_when_disabled(self, collector):
        """Test daily metrics returns error when LangSmith disabled"""
        # Mock disabled LangSmith
        with patch.object(collector, 'client', None):
            metrics = await collector.get_daily_metrics()

            assert 'error' in metrics
            assert 'LangSmith not enabled' in metrics['error']

    @pytest.mark.asyncio
    async def test_weekly_metrics_when_disabled(self, collector):
        """Test weekly metrics returns error when disabled"""
        with patch.object(collector, 'client', None):
            metrics = await collector.get_weekly_metrics()

            assert 'error' in metrics

    def test_aggregate_runs_empty(self, collector):
        """Test aggregating empty run list"""
        metrics = collector._aggregate_runs([])

        assert metrics['total_runs'] == 0
        assert metrics['successful'] == 0
        assert metrics['success_rate'] == 0.0

    def test_aggregate_runs_with_data(self, collector):
        """Test aggregating runs with data"""
        # Create mock runs
        from datetime import datetime, timedelta

        start = datetime.now()
        end = start + timedelta(seconds=2)

        mock_run1 = Mock()
        mock_run1.error = None
        mock_run1.start_time = start
        mock_run1.end_time = end
        mock_run1.outputs = {'token_usage': {'total_tokens': 100}}

        mock_run2 = Mock()
        mock_run2.error = "Error"
        mock_run2.start_time = start
        mock_run2.end_time = end
        mock_run2.outputs = {'token_usage': {'total_tokens': 50}}

        runs = [mock_run1, mock_run2]

        metrics = collector._aggregate_runs(runs)

        assert metrics['total_runs'] == 2
        assert metrics['successful'] == 1
        assert metrics['failed'] == 1
        assert metrics['success_rate'] == 50.0
        assert metrics['total_tokens'] == 150


# ============================================================================
# REALTIME METRICS TESTS
# ============================================================================

class TestRealTimeMetrics:
    """Tests for RealTimeMetrics"""

    @pytest.fixture
    def metrics(self):
        """Create real-time metrics for testing"""
        return RealTimeMetrics()

    def test_initialization(self, metrics):
        """Test metrics initialize to zero"""
        data = metrics.get_metrics()

        assert data['total_requests'] == 0
        assert data['successful'] == 0
        assert data['failed'] == 0

    def test_record_successful_request(self, metrics):
        """Test recording successful request"""
        metrics.record_request(
            success=True,
            latency=1.5,
            tokens=100,
            cost=0.005
        )

        data = metrics.get_metrics()

        assert data['total_requests'] == 1
        assert data['successful'] == 1
        assert data['failed'] == 0
        assert data['total_tokens'] == 100
        assert data['total_cost'] == 0.005

    def test_record_failed_request(self, metrics):
        """Test recording failed request"""
        metrics.record_request(
            success=False,
            latency=0.5
        )

        data = metrics.get_metrics()

        assert data['total_requests'] == 1
        assert data['successful'] == 0
        assert data['failed'] == 1

    def test_latency_calculation(self, metrics):
        """Test latency calculations"""
        # Record multiple requests
        metrics.record_request(True, 1.0)
        metrics.record_request(True, 2.0)
        metrics.record_request(True, 3.0)

        data = metrics.get_metrics()

        # Average should be 2.0
        assert data['avg_latency'] == 2.0

        # P95 should be close to 3.0
        assert data['p95_latency'] >= 2.0

    def test_reset(self, metrics):
        """Test metrics reset"""
        metrics.record_request(True, 1.0, tokens=100)
        metrics.reset()

        data = metrics.get_metrics()

        assert data['total_requests'] == 0
        assert data['total_tokens'] == 0

    def test_memory_limit(self, metrics):
        """Test latency list doesn't grow indefinitely"""
        # Record 1500 requests (should keep only last 1000)
        for i in range(1500):
            metrics.record_request(True, 1.0)

        # Should have only 1000 latencies stored
        assert len(metrics.metrics['latencies']) == 1000


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
class TestObservabilityIntegration:
    """Integration tests for observability system"""

    @pytest.mark.skipif(
        True,
        reason="Integration tests require --run-integration flag"
    )
    async def test_langsmith_setup(self):
        """Test LangSmith setup (requires valid API key)"""
        # Only runs if LANGSMITH_TRACING=true and API key is set
        if LangSmithConfig.is_enabled():
            success = setup_langsmith()
            assert success == True

    @pytest.mark.skipif(
        True,
        reason="Integration tests require --run-integration flag"
    )
    async def test_metrics_collection(self):
        """Test collecting metrics from LangSmith"""
        collector = MetricsCollector()

        if collector.client:
            metrics = await collector.get_daily_metrics()

            # Should have metrics structure
            assert 'total_runs' in metrics or 'error' in metrics


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
