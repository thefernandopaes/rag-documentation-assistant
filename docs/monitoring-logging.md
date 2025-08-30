# Monitoring & Logging

This document covers application monitoring, logging strategies, and observability practices for the DocRag application.

## Logging Architecture

### Logging Configuration

```python
import logging
import sys
from pythonjsonlogger import jsonlogger

def configure_logging():
    """Configure application logging."""
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    
    # Remove default handlers
    logger.handlers.clear()
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    
    if Config._is_production():
        # Structured JSON logging for production
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        console_handler.setFormatter(json_formatter)
    else:
        # Human-readable logging for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # Error file handler for production
    if Config._is_production():
        file_handler = logging.FileHandler('/var/log/docrag/error.log')
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    # Set external library log levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.INFO)
```

### Structured Logging

```python
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_request(self, request, response_time: float = None, 
                   status_code: int = None):
        """Log HTTP request with structured data."""
        
        log_data = {
            'event_type': 'http_request',
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent'),
            'content_length': request.content_length,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if response_time:
            log_data['response_time'] = response_time
        
        if status_code:
            log_data['status_code'] = status_code
        
        # Log level based on status code
        if status_code and status_code >= 400:
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def log_rag_operation(self, query: str, response_time: float, 
                         source_count: int, cached: bool = False):
        """Log RAG operation with metrics."""
        
        log_data = {
            'event_type': 'rag_operation',
            'query_length': len(query),
            'response_time': response_time,
            'source_count': source_count,
            'cached': cached,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: Dict = None):
        """Log error with context information."""
        
        log_data = {
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            log_data['context'] = context
        
        self.logger.error(json.dumps(log_data), exc_info=True)

# Usage throughout the application
logger = StructuredLogger(__name__)

@main_bp.route('/api/chat', methods=['POST'])
def api_chat():
    start_time = time.time()
    
    try:
        # Process request
        result = process_chat_request()
        
        # Log successful operation
        logger.log_rag_operation(
            query=request.json['query'],
            response_time=time.time() - start_time,
            source_count=len(result.get('sources', [])),
            cached=result.get('cached', False)
        )
        
        return jsonify(result)
        
    except Exception as e:
        # Log error with context
        logger.log_error(e, context={
            'endpoint': 'api_chat',
            'query_length': len(request.json.get('query', '')),
            'session_id': session.get('session_id')
        })
        raise
```

## Application Monitoring

### Health Checks

```python
@main_bp.route('/healthz')
def health_check():
    """Comprehensive health check endpoint."""
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
    
    # Database health
    try:
        db.session.execute('SELECT 1')
        health_status['checks']['database'] = 'ok'
    except Exception as e:
        health_status['checks']['database'] = f'error: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # ChromaDB health
    try:
        rag_engine = current_app.rag_engine
        collection_count = len(rag_engine.chroma_client.list_collections())
        health_status['checks']['vector_store'] = f'ok ({collection_count} collections)'
    except Exception as e:
        health_status['checks']['vector_store'] = f'error: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # OpenAI API health (optional quick check)
    try:
        # Quick API validation (cached)
        if hasattr(current_app, '_openai_health_cache'):
            health_status['checks']['openai_api'] = 'ok (cached)'
        else:
            # Minimal API test
            current_app.rag_engine.openai_client.models.list()
            health_status['checks']['openai_api'] = 'ok'
            current_app._openai_health_cache = True
    except Exception as e:
        health_status['checks']['openai_api'] = f'warning: {str(e)}'
    
    # Memory health
    memory_usage = monitor_memory_usage()
    if memory_usage['percent'] > 90:
        health_status['checks']['memory'] = f'warning: {memory_usage["percent"]:.1f}% used'
        health_status['status'] = 'degraded'
    else:
        health_status['checks']['memory'] = f'ok ({memory_usage["percent"]:.1f}% used)'
    
    # Return appropriate status code
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code

@main_bp.route('/healthz/live')
def liveness_check():
    """Simple liveness check for container orchestration."""
    return jsonify({'status': 'alive'}), 200

@main_bp.route('/healthz/ready')
def readiness_check():
    """Readiness check ensuring all dependencies are available."""
    try:
        # Quick database check
        db.session.execute('SELECT 1')
        
        # Quick ChromaDB check
        current_app.rag_engine.collection.count()
        
        return jsonify({'status': 'ready'}), 200
        
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'error': str(e)
        }), 503
```

### Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict = None):
        """Increment counter metric."""
        key = self._build_metric_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, tags: Dict = None):
        """Set gauge metric value."""
        key = self._build_metric_key(name, tags)
        self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, tags: Dict = None):
        """Record histogram metric value."""
        key = self._build_metric_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
    
    def _build_metric_key(self, name: str, tags: Dict = None) -> str:
        """Build metric key with tags."""
        if not tags:
            return name
        
        tag_string = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_string}]"
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all collected metrics."""
        
        # Process histograms
        histogram_summary = {}
        for key, values in self.histograms.items():
            if values:
                histogram_summary[key] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0]
                }
        
        return {
            'counters': self.counters,
            'gauges': self.gauges,
            'histograms': histogram_summary,
            'collection_time': datetime.utcnow().isoformat()
        }

# Global metrics collector
metrics = MetricsCollector()

# Usage in application
@main_bp.route('/api/chat', methods=['POST'])
def api_chat():
    start_time = time.time()
    
    try:
        # Process request
        result = process_request()
        
        # Record metrics
        response_time = time.time() - start_time
        metrics.increment_counter('api.requests.total', tags={'endpoint': 'chat', 'status': 'success'})
        metrics.record_histogram('api.response_time', response_time, tags={'endpoint': 'chat'})
        
        return jsonify(result)
        
    except Exception as e:
        metrics.increment_counter('api.requests.total', tags={'endpoint': 'chat', 'status': 'error'})
        raise

@main_bp.route('/metrics')
def metrics_endpoint():
    """Expose metrics for monitoring systems."""
    return jsonify(metrics.get_metrics_summary())
```

## Application Performance Monitoring (APM)

### Custom APM Implementation

```python
class SimpleAPM:
    def __init__(self):
        self.traces = []
        self.current_trace = None
    
    def start_trace(self, operation_name: str, tags: Dict = None):
        """Start a new trace."""
        trace = {
            'trace_id': str(uuid.uuid4()),
            'operation': operation_name,
            'start_time': time.time(),
            'tags': tags or {},
            'spans': []
        }
        
        self.current_trace = trace
        return trace['trace_id']
    
    def add_span(self, span_name: str, duration: float, tags: Dict = None):
        """Add span to current trace."""
        if self.current_trace:
            span = {
                'name': span_name,
                'duration': duration,
                'tags': tags or {}
            }
            self.current_trace['spans'].append(span)
    
    def end_trace(self, status: str = 'success', error: str = None):
        """End current trace."""
        if self.current_trace:
            self.current_trace.update({
                'end_time': time.time(),
                'duration': time.time() - self.current_trace['start_time'],
                'status': status,
                'error': error
            })
            
            self.traces.append(self.current_trace)
            self.current_trace = None
            
            # Keep only recent traces
            if len(self.traces) > 1000:
                self.traces = self.traces[-1000:]

# Usage
apm = SimpleAPM()

@main_bp.route('/api/chat', methods=['POST'])
def api_chat():
    trace_id = apm.start_trace('api_chat', tags={'endpoint': 'chat'})
    
    try:
        # Embedding generation span
        start_time = time.time()
        query_embedding = rag_engine._get_embedding(query)
        apm.add_span('embedding_generation', time.time() - start_time)
        
        # Vector search span
        start_time = time.time()
        search_results = rag_engine.collection.query(query_embeddings=[query_embedding])
        apm.add_span('vector_search', time.time() - start_time)
        
        # Response generation span
        start_time = time.time()
        response = rag_engine._generate_response(query, context)
        apm.add_span('response_generation', time.time() - start_time)
        
        apm.end_trace('success')
        return jsonify(response)
        
    except Exception as e:
        apm.end_trace('error', str(e))
        raise
```

## Log Levels and Categories

### Log Level Guidelines

#### DEBUG
- Detailed diagnostic information
- Variable values and state changes
- Entry/exit of functions
- **Production**: Disabled (performance impact)

```python
logger.debug(f"Processing query: {query[:50]}... (length: {len(query)})")
logger.debug(f"Retrieved {len(chunks)} chunks from vector store")
```

#### INFO
- General operational messages
- Successful operations
- Performance milestones
- **Production**: Enabled

```python
logger.info(f"Chat request processed in {response_time:.3f}s")
logger.info(f"Document ingestion completed: {doc_count} documents processed")
```

#### WARNING
- Potential issues that don't break functionality
- Performance degradation
- Recoverable errors
- **Production**: Enabled

```python
logger.warning(f"Slow response time: {response_time:.3f}s for query: {query[:30]}...")
logger.warning(f"High memory usage: {memory_percent:.1f}%")
```

#### ERROR
- Errors that break specific operations
- API failures
- Database errors
- **Production**: Enabled with alerting

```python
logger.error(f"Failed to generate embedding for query: {str(e)}")
logger.error(f"Database connection failed: {str(e)}")
```

#### CRITICAL
- Errors that may crash the application
- Security incidents
- Data corruption
- **Production**: Enabled with immediate alerting

```python
logger.critical(f"Unable to initialize RAG engine: {str(e)}")
logger.critical(f"Database migration failed: {str(e)}")
```

### Contextual Logging

```python
import contextvars

# Context variables for request tracing
request_id_context = contextvars.ContextVar('request_id')
session_id_context = contextvars.ContextVar('session_id')

class ContextualLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _get_context(self) -> Dict:
        """Get current logging context."""
        context = {}
        
        try:
            context['request_id'] = request_id_context.get()
        except LookupError:
            pass
        
        try:
            context['session_id'] = session_id_context.get()
        except LookupError:
            pass
        
        return context
    
    def info(self, message: str, **kwargs):
        """Log info with context."""
        context = self._get_context()
        log_data = {'message': message, 'context': context, **kwargs}
        self.logger.info(json.dumps(log_data))
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error with context and exception info."""
        context = self._get_context()
        log_data = {
            'message': message,
            'context': context,
            **kwargs
        }
        
        if error:
            log_data['error'] = {
                'type': type(error).__name__,
                'message': str(error)
            }
        
        self.logger.error(json.dumps(log_data), exc_info=error is not None)

# Middleware to set request context
@app.before_request
def set_request_context():
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    request_id_context.set(request_id)
    
    if 'session_id' in session:
        session_id_context.set(session['session_id'])
```

## Security Logging

### Security Event Logging

```python
class SecurityLogger:
    def __init__(self):
        self.logger = logging.getLogger('security')
        
        # Separate handler for security events
        if Config._is_production():
            security_handler = logging.FileHandler('/var/log/docrag/security.log')
            security_formatter = jsonlogger.JsonFormatter()
            security_handler.setFormatter(security_formatter)
            self.logger.addHandler(security_handler)
    
    def log_authentication_attempt(self, ip_address: str, success: bool, 
                                  endpoint: str = None):
        """Log authentication attempts."""
        event_data = {
            'event_type': 'authentication',
            'ip_address': ip_address,
            'success': success,
            'endpoint': endpoint,
            'timestamp': datetime.utcnow().isoformat(),
            'user_agent': request.headers.get('User-Agent', 'unknown')
        }
        
        if success:
            self.logger.info(json.dumps(event_data))
        else:
            self.logger.warning(json.dumps(event_data))
    
    def log_rate_limit_violation(self, ip_address: str, endpoint: str, 
                                current_count: int, limit: int):
        """Log rate limit violations."""
        event_data = {
            'event_type': 'rate_limit_violation',
            'ip_address': ip_address,
            'endpoint': endpoint,
            'current_count': current_count,
            'limit': limit,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.logger.warning(json.dumps(event_data))
    
    def log_suspicious_activity(self, ip_address: str, activity_type: str, 
                               details: Dict):
        """Log suspicious activities."""
        event_data = {
            'event_type': 'suspicious_activity',
            'activity_type': activity_type,
            'ip_address': ip_address,
            'details': details,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'high'
        }
        
        self.logger.error(json.dumps(event_data))

# Usage
security_logger = SecurityLogger()

def validate_admin_key(auth_header: str, ip_address: str) -> bool:
    """Validate admin key with security logging."""
    
    if not auth_header or not auth_header.startswith('Bearer '):
        security_logger.log_authentication_attempt(
            ip_address=ip_address,
            success=False,
            endpoint='admin_api'
        )
        return False
    
    provided_key = auth_header[7:]
    expected_key = Config.ADMIN_API_KEY
    
    if not expected_key or not hmac.compare_digest(provided_key, expected_key):
        security_logger.log_authentication_attempt(
            ip_address=ip_address,
            success=False,
            endpoint='admin_api'
        )
        return False
    
    security_logger.log_authentication_attempt(
        ip_address=ip_address,
        success=True,
        endpoint='admin_api'
    )
    return True
```

## Performance Monitoring

### Real-Time Performance Tracking

```python
class PerformanceTracker:
    def __init__(self):
        self.request_times = {}
        self.error_counts = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def track_request(self, endpoint: str, duration: float, status_code: int):
        """Track request performance."""
        
        if endpoint not in self.request_times:
            self.request_times[endpoint] = []
        
        self.request_times[endpoint].append({
            'duration': duration,
            'status_code': status_code,
            'timestamp': time.time()
        })
        
        # Keep only recent data (last hour)
        cutoff_time = time.time() - 3600
        self.request_times[endpoint] = [
            req for req in self.request_times[endpoint]
            if req['timestamp'] > cutoff_time
        ]
        
        # Track errors
        if status_code >= 400:
            error_key = f"{endpoint}:{status_code}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_performance_summary(self, endpoint: str = None) -> Dict:
        """Get performance summary for endpoint or all endpoints."""
        
        if endpoint:
            return self._get_endpoint_summary(endpoint)
        
        # Summary for all endpoints
        summary = {}
        for ep in self.request_times.keys():
            summary[ep] = self._get_endpoint_summary(ep)
        
        return summary
    
    def _get_endpoint_summary(self, endpoint: str) -> Dict:
        """Get performance summary for specific endpoint."""
        
        requests = self.request_times.get(endpoint, [])
        if not requests:
            return {'error': 'No data available'}
        
        durations = [req['duration'] for req in requests]
        status_codes = [req['status_code'] for req in requests]
        
        return {
            'total_requests': len(requests),
            'avg_response_time': sum(durations) / len(durations),
            'min_response_time': min(durations),
            'max_response_time': max(durations),
            'p95_response_time': sorted(durations)[int(len(durations) * 0.95)],
            'error_rate': sum(1 for sc in status_codes if sc >= 400) / len(status_codes),
            'requests_per_minute': len(requests) / 60  # Approximate
        }

# Global performance tracker
perf_tracker = PerformanceTracker()

@app.after_request
def track_request_performance(response):
    """Track request performance automatically."""
    
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        endpoint = request.endpoint or 'unknown'
        
        perf_tracker.track_request(
            endpoint=endpoint,
            duration=duration,
            status_code=response.status_code
        )
        
        # Add performance header
        response.headers['X-Response-Time'] = f"{duration*1000:.2f}ms"
    
    return response
```

### Resource Monitoring

```python
import psutil
import threading

class ResourceMonitor:
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.metrics_history = []
        self.monitoring = False
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 24 hours of data
                cutoff_time = time.time() - 86400
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m['timestamp'] > cutoff_time
                ]
                
                # Alert on resource issues
                self._check_resource_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
            
            time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> Dict:
        """Collect current system metrics."""
        process = psutil.Process()
        
        return {
            'timestamp': time.time(),
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'open_files': len(process.open_files()),
            'connections': len(process.connections()),
            'threads': process.num_threads()
        }
    
    def _check_resource_alerts(self, metrics: Dict):
        """Check for resource usage alerts."""
        
        # Memory alert
        if metrics['memory_percent'] > 85:
            logger.warning(f"High memory usage: {metrics['memory_percent']:.1f}%")
        
        # CPU alert
        if metrics['cpu_percent'] > 80:
            logger.warning(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        # File descriptor alert
        if metrics['open_files'] > 1000:
            logger.warning(f"High file descriptor usage: {metrics['open_files']}")

# Start resource monitoring
resource_monitor = ResourceMonitor()
resource_monitor.start_monitoring()
```

## External Monitoring Integration

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
request_count = Counter('docrag_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('docrag_request_duration_seconds', 'Request duration', ['endpoint'])
memory_usage = Gauge('docrag_memory_usage_bytes', 'Memory usage in bytes')
active_sessions = Gauge('docrag_active_sessions', 'Number of active sessions')

@app.before_request
def prometheus_start_timer():
    request.start_time = time.time()

@app.after_request
def prometheus_record_metrics(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        
        # Record metrics
        request_count.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            endpoint=request.endpoint or 'unknown'
        ).observe(duration)
        
        # Update resource metrics
        process = psutil.Process()
        memory_usage.set(process.memory_info().rss)
    
    return response

@app.route('/metrics')
def prometheus_metrics():
    """Expose Prometheus metrics."""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
```

### Log Aggregation

#### ELK Stack Integration

```python
import logging
from pythonjsonlogger import jsonlogger
import socket

def configure_elk_logging():
    """Configure logging for ELK stack integration."""
    
    # Create custom formatter with additional fields
    class ELKFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            
            # Add application metadata
            log_record['application'] = 'docrag'
            log_record['hostname'] = socket.gethostname()
            log_record['environment'] = Config.ENV
            
            # Add request context if available
            try:
                log_record['request_id'] = request_id_context.get()
                log_record['session_id'] = session_id_context.get()
            except LookupError:
                pass
    
    # Configure handler
    elk_handler = logging.StreamHandler()
    elk_handler.setFormatter(ELKFormatter())
    
    # Add to root logger
    logging.getLogger().addHandler(elk_handler)
```

#### Fluentd Integration

```python
from fluent import sender
from fluent.handler import FluentHandler

def configure_fluentd_logging():
    """Configure Fluentd logging integration."""
    
    # Create Fluentd handler
    fluentd_handler = FluentHandler('docrag', host='localhost', port=24224)
    
    # Custom formatter for Fluentd
    class FluentdFormatter(logging.Formatter):
        def format(self, record):
            return {
                'level': record.levelname,
                'message': record.getMessage(),
                'timestamp': record.created,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
    
    fluentd_handler.setFormatter(FluentdFormatter())
    
    # Add to application logger
    app_logger = logging.getLogger('docrag')
    app_logger.addHandler(fluentd_handler)
```

## Alerting and Notifications

### Alert Configuration

```python
class AlertManager:
    def __init__(self):
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'response_time': 5.0,  # 5 seconds
            'memory_usage': 0.85,  # 85% memory usage
            'db_connections': 0.8   # 80% of pool used
        }
        self.alert_cooldown = {}  # Prevent spam
        self.notification_channels = []
    
    def check_and_alert(self, metric_name: str, current_value: float, 
                       context: Dict = None):
        """Check metric against threshold and send alert if needed."""
        
        threshold = self.alert_thresholds.get(metric_name)
        if not threshold:
            return
        
        # Check if alert condition is met
        alert_triggered = current_value > threshold
        
        if alert_triggered:
            # Check cooldown to prevent spam
            cooldown_key = f"{metric_name}:{context.get('endpoint', 'global')}"
            last_alert_time = self.alert_cooldown.get(cooldown_key, 0)
            
            if time.time() - last_alert_time > 300:  # 5 minute cooldown
                self._send_alert(metric_name, current_value, threshold, context)
                self.alert_cooldown[cooldown_key] = time.time()
    
    def _send_alert(self, metric_name: str, value: float, 
                   threshold: float, context: Dict = None):
        """Send alert notification."""
        
        alert_message = {
            'alert_type': 'performance_threshold_exceeded',
            'metric': metric_name,
            'current_value': value,
            'threshold': threshold,
            'severity': self._calculate_severity(value, threshold),
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat(),
            'hostname': socket.gethostname()
        }
        
        # Log alert
        logger.error(f"ALERT: {json.dumps(alert_message)}")
        
        # Send to notification channels
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert_message)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _calculate_severity(self, value: float, threshold: float) -> str:
        """Calculate alert severity based on threshold exceedance."""
        ratio = value / threshold
        
        if ratio > 2.0:
            return 'critical'
        elif ratio > 1.5:
            return 'high'
        elif ratio > 1.2:
            return 'medium'
        else:
            return 'low'

# Global alert manager
alert_manager = AlertManager()

# Usage in performance monitoring
def monitor_and_alert():
    """Monitor performance and trigger alerts."""
    
    # Check response time
    avg_response_time = perf_tracker.get_avg_response_time()
    alert_manager.check_and_alert('response_time', avg_response_time)
    
    # Check error rate
    error_rate = perf_tracker.get_error_rate()
    alert_manager.check_and_alert('error_rate', error_rate)
    
    # Check memory usage
    memory_percent = psutil.Process().memory_percent()
    alert_manager.check_and_alert('memory_usage', memory_percent / 100)
```

### Notification Channels

```python
class SlackNotificationChannel:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert_data: Dict):
        """Send alert to Slack."""
        
        severity_colors = {
            'low': '#36a64f',      # Green
            'medium': '#ff9900',   # Orange  
            'high': '#ff0000',     # Red
            'critical': '#800080'   # Purple
        }
        
        message = {
            'text': f"DocRag Alert: {alert_data['metric']} threshold exceeded",
            'attachments': [{
                'color': severity_colors.get(alert_data['severity'], '#ff9900'),
                'fields': [
                    {
                        'title': 'Metric',
                        'value': alert_data['metric'],
                        'short': True
                    },
                    {
                        'title': 'Current Value',
                        'value': f"{alert_data['current_value']:.3f}",
                        'short': True
                    },
                    {
                        'title': 'Threshold',
                        'value': f"{alert_data['threshold']:.3f}",
                        'short': True
                    },
                    {
                        'title': 'Severity',
                        'value': alert_data['severity'].title(),
                        'short': True
                    }
                ],
                'timestamp': int(time.time())
            }]
        }
        
        requests.post(self.webhook_url, json=message)

# Configure notification channels
if Config.SLACK_WEBHOOK_URL:
    slack_channel = SlackNotificationChannel(Config.SLACK_WEBHOOK_URL)
    alert_manager.notification_channels.append(slack_channel)
```

## Log Analysis and Troubleshooting

### Log Analysis Tools

```python
def analyze_error_patterns(log_file_path: str) -> Dict:
    """Analyze error patterns in log files."""
    
    error_patterns = {}
    total_errors = 0
    
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                
                if log_entry.get('levelname') == 'ERROR':
                    total_errors += 1
                    error_type = log_entry.get('error', {}).get('type', 'unknown')
                    
                    if error_type not in error_patterns:
                        error_patterns[error_type] = {
                            'count': 0,
                            'examples': []
                        }
                    
                    error_patterns[error_type]['count'] += 1
                    
                    # Keep sample error messages
                    if len(error_patterns[error_type]['examples']) < 3:
                        error_patterns[error_type]['examples'].append(
                            log_entry.get('message', '')
                        )
                        
            except json.JSONDecodeError:
                continue  # Skip non-JSON log lines
    
    # Calculate percentages
    for error_type in error_patterns:
        error_patterns[error_type]['percentage'] = (
            error_patterns[error_type]['count'] / total_errors * 100
        )
    
    return {
        'total_errors': total_errors,
        'error_patterns': error_patterns,
        'analysis_time': datetime.utcnow().isoformat()
    }

def get_slow_requests(log_file_path: str, threshold_ms: float = 5000) -> List[Dict]:
    """Find slow requests from logs."""
    
    slow_requests = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                
                if (log_entry.get('event_type') == 'http_request' and
                    log_entry.get('response_time', 0) > threshold_ms):
                    
                    slow_requests.append({
                        'path': log_entry.get('path'),
                        'method': log_entry.get('method'),
                        'response_time': log_entry.get('response_time'),
                        'timestamp': log_entry.get('timestamp'),
                        'remote_addr': log_entry.get('remote_addr')
                    })
                    
            except json.JSONDecodeError:
                continue
    
    # Sort by response time (slowest first)
    slow_requests.sort(key=lambda x: x['response_time'], reverse=True)
    
    return slow_requests
```

---

*Effective monitoring and logging are essential for maintaining a production-ready application. Implement these practices gradually and adjust based on your operational needs.*