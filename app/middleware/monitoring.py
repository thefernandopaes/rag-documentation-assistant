"""
API-Specific Monitoring and Analytics Module for DocRag System.

This module provides comprehensive monitoring capabilities specifically designed
for API documentation assistance, including usage patterns, performance metrics,
and API-specific analytics.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

from flask import request, g
from sqlalchemy import text
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class APIQueryMetrics:
    """Metrics for API-related queries."""
    query_id: str
    user_query: str
    query_type: str  # 'api_endpoint', 'authentication', 'parameters', 'examples', 'general'
    api_domain: Optional[str]
    detected_api: Optional[str]
    response_time: float
    token_usage: int
    cache_hit: bool
    code_examples_generated: int
    endpoints_referenced: int
    timestamp: datetime
    user_feedback: Optional[int] = None


@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    cache_size: int
    cache_hit_rate: float
    error_rate: float
    avg_response_time: float


class APIMonitoringSystem:
    """Comprehensive monitoring system for API documentation RAG."""
    
    def __init__(self, db_session=None):
        self.db_session = db_session
        self.query_metrics: deque = deque(maxlen=10000)  # In-memory metrics buffer
        self.api_usage_stats = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.performance_buffer = deque(maxlen=1000)
        
        # Rate limiting tracking
        self.rate_limit_violations = defaultdict(list)
        
        # API discovery tracking
        self.discovered_apis = {}
        self.discovery_stats = {
            'total_discoveries': 0,
            'successful_discoveries': 0,
            'failed_discoveries': 0,
            'last_discovery': None
        }
        
    def track_api_query(self, query_data: Dict[str, Any]) -> str:
        """Track an API-related query with comprehensive metrics."""
        try:
            # Generate unique query ID
            query_id = f"q_{int(time.time() * 1000)}_{hash(query_data.get('query', '')) % 10000}"
            
            # Analyze query type
            query_type = self._classify_query_type(query_data.get('query', ''))
            
            # Extract API information
            api_info = self._extract_api_info(query_data)
            
            # Create metrics object
            metrics = APIQueryMetrics(
                query_id=query_id,
                user_query=query_data.get('query', ''),
                query_type=query_type,
                api_domain=api_info.get('domain'),
                detected_api=api_info.get('api_name'),
                response_time=query_data.get('response_time', 0.0),
                token_usage=query_data.get('token_usage', 0),
                cache_hit=query_data.get('cached', False),
                code_examples_generated=len(query_data.get('examples', [])),
                endpoints_referenced=len(query_data.get('endpoints', [])),
                timestamp=datetime.utcnow()
            )
            
            # Store metrics
            self.query_metrics.append(metrics)
            
            # Update aggregated stats
            self._update_usage_stats(metrics)
            
            # Store in database if available
            if Config.MONITORING_ENABLED and self.db_session:
                self._store_metrics_to_db(metrics)
                
            logger.info(f"Tracked API query: {query_id} - Type: {query_type}")
            return query_id
            
        except Exception as e:
            logger.error(f"Error tracking API query: {e}")
            return ""
    
    def track_api_discovery(self, base_url: str, discovery_result: Dict[str, Any]):
        """Track API discovery attempts and results."""
        try:
            self.discovery_stats['total_discoveries'] += 1
            self.discovery_stats['last_discovery'] = datetime.utcnow().isoformat()
            
            if discovery_result.get('success', False):
                self.discovery_stats['successful_discoveries'] += 1
                
                # Store discovered API information
                api_sources = discovery_result.get('sources', [])
                if api_sources:
                    self.discovered_apis[base_url] = {
                        'sources': api_sources,
                        'discovered_at': datetime.utcnow().isoformat(),
                        'source_count': len(api_sources),
                        'types': list(set(source.doc_type for source in api_sources))
                    }
                    
                logger.info(f"Successfully discovered {len(api_sources)} API sources from {base_url}")
            else:
                self.discovery_stats['failed_discoveries'] += 1
                error = discovery_result.get('error', 'Unknown error')
                self.error_counts[f"discovery_error_{error}"] += 1
                logger.warning(f"Failed to discover APIs from {base_url}: {error}")
                
        except Exception as e:
            logger.error(f"Error tracking API discovery: {e}")
    
    def track_rate_limit_violation(self, client_id: str):
        """Track rate limit violations for monitoring."""
        now = datetime.utcnow()
        self.rate_limit_violations[client_id].append(now)
        
        # Clean old violations (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self.rate_limit_violations[client_id] = [
            timestamp for timestamp in self.rate_limit_violations[client_id]
            if timestamp > cutoff
        ]
    
    def track_error(self, error_type: str, error_details: Dict[str, Any]):
        """Track system errors for monitoring."""
        self.error_counts[error_type] += 1
        
        if Config.ERROR_REPORTING_ENABLED:
            error_data = {
                'type': error_type,
                'timestamp': datetime.utcnow().isoformat(),
                'details': error_details,
                'request_path': getattr(request, 'path', 'unknown'),
                'user_agent': getattr(request, 'user_agent', 'unknown')
            }
            
            logger.error(f"Tracked error: {error_type} - {error_details}")
    
    def get_api_usage_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive API usage analytics."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Filter recent metrics
            recent_metrics = [
                m for m in self.query_metrics 
                if m.timestamp > cutoff_date
            ]
            
            if not recent_metrics:
                return self._empty_analytics()
            
            # Calculate analytics
            total_queries = len(recent_metrics)
            avg_response_time = sum(m.response_time for m in recent_metrics) / total_queries
            cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / total_queries
            
            # Query type distribution
            query_types = defaultdict(int)
            for m in recent_metrics:
                query_types[m.query_type] += 1
            
            # API popularity
            api_popularity = defaultdict(int)
            for m in recent_metrics:
                if m.detected_api:
                    api_popularity[m.detected_api] += 1
            
            # Performance trends
            daily_metrics = self._group_metrics_by_day(recent_metrics)
            
            # Code generation stats
            total_examples = sum(m.code_examples_generated for m in recent_metrics)
            total_endpoints = sum(m.endpoints_referenced for m in recent_metrics)
            
            return {
                'period_days': days,
                'total_queries': total_queries,
                'avg_response_time': round(avg_response_time, 3),
                'cache_hit_rate': round(cache_hit_rate, 3),
                'query_type_distribution': dict(query_types),
                'api_popularity': dict(sorted(api_popularity.items(), key=lambda x: x[1], reverse=True)[:10]),
                'daily_metrics': daily_metrics,
                'code_generation': {
                    'total_examples_generated': total_examples,
                    'total_endpoints_referenced': total_endpoints,
                    'avg_examples_per_query': round(total_examples / total_queries, 2) if total_queries > 0 else 0
                },
                'discovery_stats': self.discovery_stats.copy(),
                'discovered_apis_count': len(self.discovered_apis),
                'error_summary': dict(list(self.error_counts.items())[-10:])  # Last 10 error types
            }
            
        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            return self._empty_analytics()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            import psutil
            
            # System resources
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Application-specific metrics
            recent_queries = len([m for m in self.query_metrics if m.timestamp > datetime.utcnow() - timedelta(minutes=5)])
            recent_errors = sum(1 for errors in self.error_counts.values())
            
            cache_stats = self._get_cache_stats()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'system': {
                    'cpu_usage_percent': cpu_usage,
                    'memory_usage_percent': memory.percent,
                    'memory_available_gb': round(memory.available / (1024**3), 2)
                },
                'application': {
                    'recent_queries_5min': recent_queries,
                    'total_queries_buffered': len(self.query_metrics),
                    'recent_errors': recent_errors,
                    'rate_limit_violations': sum(len(violations) for violations in self.rate_limit_violations.values())
                },
                'cache': cache_stats,
                'discovery': {
                    'total_discovered_apis': len(self.discovered_apis),
                    'last_discovery': self.discovery_stats.get('last_discovery'),
                    'success_rate': (
                        self.discovery_stats['successful_discoveries'] / max(self.discovery_stats['total_discoveries'], 1)
                    )
                }
            }
            
        except ImportError:
            logger.warning("psutil not available, providing limited system health metrics")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'system': {'status': 'monitoring_limited'},
                'application': {
                    'total_queries_buffered': len(self.query_metrics),
                    'total_discovered_apis': len(self.discovered_apis)
                }
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'timestamp': datetime.utcnow().isoformat(), 'status': 'error', 'message': str(e)}
    
    def cleanup_old_data(self):
        """Clean up old monitoring data based on retention policy."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=Config.ANALYTICS_RETENTION_DAYS)
            
            # Clean in-memory metrics
            initial_count = len(self.query_metrics)
            self.query_metrics = deque(
                (m for m in self.query_metrics if m.timestamp > cutoff_date),
                maxlen=10000
            )
            cleaned_count = initial_count - len(self.query_metrics)
            
            # Clean rate limit violations
            for client_id in list(self.rate_limit_violations.keys()):
                self.rate_limit_violations[client_id] = [
                    timestamp for timestamp in self.rate_limit_violations[client_id]
                    if timestamp > cutoff_date
                ]
                if not self.rate_limit_violations[client_id]:
                    del self.rate_limit_violations[client_id]
            
            logger.info(f"Cleaned {cleaned_count} old metrics records")
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of API query."""
        query_lower = query.lower()
        
        # API endpoint queries
        if any(keyword in query_lower for keyword in ['endpoint', 'get ', 'post ', 'put ', 'delete ', 'patch ', 'api call']):
            return 'api_endpoint'
        
        # Authentication queries
        if any(keyword in query_lower for keyword in ['auth', 'token', 'key', 'login', 'bearer', 'oauth']):
            return 'authentication'
        
        # Parameter queries
        if any(keyword in query_lower for keyword in ['parameter', 'param', 'field', 'body', 'header']):
            return 'parameters'
        
        # Code example queries
        if any(keyword in query_lower for keyword in ['example', 'curl', 'python', 'javascript', 'code', 'sample']):
            return 'examples'
        
        return 'general'
    
    def _extract_api_info(self, query_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Extract API information from query data."""
        query = query_data.get('query', '').lower()
        
        # Try to detect API from query content
        api_indicators = {
            'stripe': ['stripe', 'payment', 'charge', 'customer'],
            'github': ['github', 'repository', 'commit', 'pull request'],
            'openai': ['openai', 'gpt', 'completion', 'embedding'],
            'twilio': ['twilio', 'sms', 'call', 'phone'],
            'discord': ['discord', 'bot', 'guild', 'channel']
        }
        
        detected_api = None
        for api_name, indicators in api_indicators.items():
            if any(indicator in query for indicator in indicators):
                detected_api = api_name
                break
        
        # Extract domain if URL is mentioned
        import re
        url_pattern = r'https?://([a-zA-Z0-9.-]+)'
        urls = re.findall(url_pattern, query)
        domain = urls[0] if urls else None
        
        return {
            'api_name': detected_api,
            'domain': domain
        }
    
    def _update_usage_stats(self, metrics: APIQueryMetrics):
        """Update aggregated usage statistics."""
        self.api_usage_stats['total_queries'] += 1
        self.api_usage_stats[f'query_type_{metrics.query_type}'] += 1
        
        if metrics.detected_api:
            self.api_usage_stats[f'api_{metrics.detected_api}'] += 1
        
        if metrics.cache_hit:
            self.api_usage_stats['cache_hits'] += 1
        
        self.api_usage_stats['total_response_time'] += metrics.response_time
        self.api_usage_stats['total_tokens'] += metrics.token_usage
        self.api_usage_stats['total_examples'] += metrics.code_examples_generated
    
    def _store_metrics_to_db(self, metrics: APIQueryMetrics):
        """Store metrics to database if available."""
        try:
            if self.db_session:
                # Convert to dict for JSON storage
                metrics_dict = asdict(metrics)
                metrics_dict['timestamp'] = metrics.timestamp.isoformat()
                
                # Store in analytics table (would need to be created)
                query = text("""
                    INSERT INTO api_analytics 
                    (query_id, metrics_data, created_at) 
                    VALUES (:query_id, :metrics_data, :created_at)
                """)
                
                self.db_session.execute(query, {
                    'query_id': metrics.query_id,
                    'metrics_data': json.dumps(metrics_dict),
                    'created_at': metrics.timestamp
                })
                self.db_session.commit()
                
        except Exception as e:
            logger.error(f"Error storing metrics to database: {e}")
    
    def _group_metrics_by_day(self, metrics: List[APIQueryMetrics]) -> Dict[str, Dict[str, Any]]:
        """Group metrics by day for trend analysis."""
        daily_groups = defaultdict(list)
        
        for metric in metrics:
            day_key = metric.timestamp.date().isoformat()
            daily_groups[day_key].append(metric)
        
        daily_metrics = {}
        for day, day_metrics in daily_groups.items():
            daily_metrics[day] = {
                'total_queries': len(day_metrics),
                'avg_response_time': sum(m.response_time for m in day_metrics) / len(day_metrics),
                'cache_hit_rate': sum(1 for m in day_metrics if m.cache_hit) / len(day_metrics),
                'examples_generated': sum(m.code_examples_generated for m in day_metrics)
            }
        
        return daily_metrics
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics if available."""
        try:
            # This would integrate with the actual cache system
            total_cached = sum(1 for m in self.query_metrics if m.cache_hit)
            total_queries = len(self.query_metrics)
            
            return {
                'hit_rate': round(total_cached / max(total_queries, 1), 3),
                'total_cached_responses': total_cached,
                'estimated_size_mb': 'unknown'  # Would need cache system integration
            }
        except Exception:
            return {'status': 'unavailable'}
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure."""
        return {
            'period_days': 0,
            'total_queries': 0,
            'avg_response_time': 0,
            'cache_hit_rate': 0,
            'query_type_distribution': {},
            'api_popularity': {},
            'daily_metrics': {},
            'code_generation': {
                'total_examples_generated': 0,
                'total_endpoints_referenced': 0,
                'avg_examples_per_query': 0
            },
            'discovery_stats': self.discovery_stats.copy(),
            'discovered_apis_count': 0,
            'error_summary': {}
        }


# Global monitoring instance
monitoring = APIMonitoringSystem()


def init_monitoring(db_session=None):
    """Initialize monitoring system with database session."""
    global monitoring
    monitoring.db_session = db_session
    logger.info("API monitoring system initialized")


def track_request_start():
    """Track the start of a request for timing."""
    g.request_start_time = time.time()


def track_request_end():
    """Track the end of a request for performance monitoring."""
    if hasattr(g, 'request_start_time'):
        response_time = time.time() - g.request_start_time
        monitoring.performance_buffer.append({
            'path': request.path,
            'method': request.method,
            'response_time': response_time,
            'timestamp': datetime.utcnow()
        })