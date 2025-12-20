"""
Metrics Collection - Aggregate Metrics from LangSmith

Collects and aggregates metrics from LangSmith for monitoring and analysis.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collect and aggregate metrics from LangSmith.

    Provides methods to fetch and analyze runs from LangSmith for monitoring.
    """

    def __init__(self):
        """Initialize metrics collector"""
        from langsmith_config import LangSmithConfig

        self.config = LangSmithConfig
        self.client = LangSmithConfig.get_client()

        if not self.client:
            logger.warning("Metrics collector initialized without LangSmith client (tracing disabled)")

    async def get_daily_metrics(self) -> Dict:
        """
        Get metrics for last 24 hours.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.client:
            return {
                'error': 'LangSmith not enabled',
                'message': 'Set LANGSMITH_TRACING=true to enable metrics'
            }

        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)

            # Get runs from last 24h
            project_name = os.getenv("LANGSMITH_PROJECT", "rag-assistant")

            runs = list(self.client.list_runs(
                project_name=project_name,
                start_time=start_time,
                end_time=end_time
            ))

            # Aggregate metrics
            metrics = self._aggregate_runs(runs)
            metrics['period'] = '24h'
            metrics['start_time'] = start_time.isoformat()
            metrics['end_time'] = end_time.isoformat()

            logger.info(f"Daily metrics collected: {metrics['total_runs']} runs")

            return metrics

        except Exception as e:
            logger.error(f"Error getting daily metrics: {e}", exc_info=True)
            return {
                'error': str(e),
                'message': 'Failed to fetch metrics from LangSmith'
            }

    async def get_weekly_metrics(self) -> Dict:
        """
        Get metrics for last 7 days.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.client:
            return {
                'error': 'LangSmith not enabled',
                'message': 'Set LANGSMITH_TRACING=true to enable metrics'
            }

        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)

            project_name = os.getenv("LANGSMITH_PROJECT", "rag-assistant")

            runs = list(self.client.list_runs(
                project_name=project_name,
                start_time=start_time,
                end_time=end_time
            ))

            metrics = self._aggregate_runs(runs)
            metrics['period'] = '7d'
            metrics['start_time'] = start_time.isoformat()
            metrics['end_time'] = end_time.isoformat()

            logger.info(f"Weekly metrics collected: {metrics['total_runs']} runs")

            return metrics

        except Exception as e:
            logger.error(f"Error getting weekly metrics: {e}", exc_info=True)
            return {
                'error': str(e),
                'message': 'Failed to fetch metrics from LangSmith'
            }

    async def get_run_details(self, run_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Run details or None if not found
        """
        if not self.client:
            return None

        try:
            run = self.client.read_run(run_id)

            return {
                'id': str(run.id),
                'name': run.name,
                'start_time': run.start_time.isoformat() if run.start_time else None,
                'end_time': run.end_time.isoformat() if run.end_time else None,
                'status': 'success' if not run.error else 'error',
                'error': run.error if run.error else None,
                'inputs': run.inputs,
                'outputs': run.outputs,
                'latency': (
                    (run.end_time - run.start_time).total_seconds()
                    if run.end_time and run.start_time else None
                )
            }

        except Exception as e:
            logger.error(f"Error getting run details: {e}")
            return None

    def _aggregate_runs(self, runs: List) -> Dict:
        """
        Aggregate metrics from list of runs.

        Args:
            runs: List of LangSmith run objects

        Returns:
            Aggregated metrics dictionary
        """
        if not runs:
            return {
                'total_runs': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0,
                'total_tokens': 0,
                'avg_latency': 0.0,
                'p50_latency': 0.0,
                'p95_latency': 0.0,
                'estimated_cost': 0.0
            }

        # Count successful/failed
        total_runs = len(runs)
        successful = sum(1 for r in runs if not r.error)
        failed = total_runs - successful

        # Calculate latencies
        latencies = []
        for r in runs:
            if r.end_time and r.start_time:
                latency = (r.end_time - r.start_time).total_seconds()
                latencies.append(latency)

        latencies.sort()

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate percentiles
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = int(len(latencies) * 0.95)
        p50_latency = latencies[p50_idx] if latencies else 0.0
        p95_latency = latencies[p95_idx] if latencies else 0.0

        # Count tokens
        total_tokens = 0
        for r in runs:
            if r.outputs and isinstance(r.outputs, dict):
                token_usage = r.outputs.get('token_usage', {})
                total_tokens += token_usage.get('total_tokens', 0)

        # Estimate cost (rough estimate: $0.04 per 1K tokens average)
        estimated_cost = (total_tokens * 0.04) / 1000

        return {
            'total_runs': total_runs,
            'successful': successful,
            'failed': failed,
            'success_rate': round((successful / total_runs * 100), 2) if total_runs > 0 else 0.0,
            'total_tokens': total_tokens,
            'avg_latency': round(avg_latency, 2),
            'p50_latency': round(p50_latency, 2),
            'p95_latency': round(p95_latency, 2),
            'estimated_cost': round(estimated_cost, 4),
            'latency_distribution': {
                'min': round(min(latencies), 2) if latencies else 0.0,
                'max': round(max(latencies), 2) if latencies else 0.0,
                'median': round(p50_latency, 2)
            }
        }


class RealTimeMetrics:
    """
    Real-time metrics tracking (in-memory).

    Useful for tracking metrics without LangSmith or for immediate feedback.
    """

    def __init__(self):
        """Initialize real-time metrics"""
        self.metrics = {
            'requests': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'latencies': []
        }

    def record_request(
        self,
        success: bool,
        latency: float,
        tokens: int = 0,
        cost: float = 0.0
    ) -> None:
        """
        Record a request.

        Args:
            success: Whether request succeeded
            latency: Request latency in seconds
            tokens: Tokens used
            cost: Cost in USD
        """
        self.metrics['requests'] += 1

        if success:
            self.metrics['successful'] += 1
        else:
            self.metrics['failed'] += 1

        self.metrics['total_tokens'] += tokens
        self.metrics['total_cost'] += cost
        self.metrics['latencies'].append(latency)

        # Keep only last 1000 latencies to prevent memory issues
        if len(self.metrics['latencies']) > 1000:
            self.metrics['latencies'] = self.metrics['latencies'][-1000:]

    def get_metrics(self) -> Dict:
        """
        Get current metrics.

        Returns:
            Current metrics dictionary
        """
        latencies = self.metrics['latencies']

        if not latencies:
            avg_latency = 0.0
            p95_latency = 0.0
        else:
            sorted_latencies = sorted(latencies)
            avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_idx]

        return {
            'total_requests': self.metrics['requests'],
            'successful': self.metrics['successful'],
            'failed': self.metrics['failed'],
            'success_rate': (
                round((self.metrics['successful'] / self.metrics['requests'] * 100), 2)
                if self.metrics['requests'] > 0 else 0.0
            ),
            'total_tokens': self.metrics['total_tokens'],
            'total_cost': round(self.metrics['total_cost'], 4),
            'avg_latency': round(avg_latency, 2),
            'p95_latency': round(p95_latency, 2)
        }

    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics = {
            'requests': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'latencies': []
        }


# Global instance for real-time metrics
_realtime_metrics = RealTimeMetrics()


def get_realtime_metrics() -> RealTimeMetrics:
    """
    Get global real-time metrics instance.

    Returns:
        RealTimeMetrics instance
    """
    return _realtime_metrics
