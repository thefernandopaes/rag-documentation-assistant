# API Design

This document describes the REST API design principles, endpoints, request/response formats, and conventions used in the DocRag application.

## API Design Principles

### RESTful Design
- **Resource-based URLs**: `/api/conversations` (not `/api/getConversations`)
- **HTTP methods**: GET for retrieval, POST for creation, PUT for updates, DELETE for removal
- **Status codes**: Meaningful HTTP status codes for different scenarios
- **Stateless**: Each request contains all necessary information

### Response Format
- **JSON**: All responses use JSON format
- **Consistent structure**: Standard response envelope
- **Error handling**: Consistent error response format
- **Metadata**: Include response metadata (timing, pagination, etc.)

### Versioning Strategy
- **URL versioning**: `/api/v1/endpoint` (future consideration)
- **Header versioning**: `Accept: application/vnd.docrag.v1+json` (alternative)
- **Backward compatibility**: Maintain compatibility when possible

## API Endpoints

### Core Endpoints

#### Chat Interface

**POST /api/chat**
```http
POST /api/chat
Content-Type: application/json
```

Request:
```json
{
    "query": "What is React?",
    "context": "I'm learning frontend development"
}
```

Response:
```json
{
    "response": "React is a JavaScript library for building user interfaces...",
    "sources": [
        "https://react.dev/learn/describing-the-ui",
        "https://react.dev/learn/your-first-component"
    ],
    "response_time": 1.234,
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Error Response:
```json
{
    "error": "Query is required",
    "code": "INVALID_REQUEST",
    "details": {
        "field": "query",
        "message": "Query cannot be empty"
    }
}
```

#### Statistics

**GET /api/stats**
```http
GET /api/stats
```

Response:
```json
{
    "stats": {
        "total_documents": 1547,
        "total_conversations": 892,
        "cache_stats": {
            "hits": 456,
            "misses": 123,
            "hit_rate": 0.787
        },
        "vector_store": {
            "collection_count": 1,
            "document_count": 1547
        }
    },
    "timestamp": "2024-08-10T19:48:00Z"
}
```

#### Health Check

**GET /healthz**
```http
GET /healthz
```

Response:
```json
{
    "status": "healthy",
    "checks": {
        "database": "ok",
        "vector_store": "ok",
        "openai_api": "ok"
    },
    "timestamp": "2024-08-10T19:48:00Z",
    "version": "1.0.0"
}
```

### Administrative Endpoints

#### Initialize System

**POST /api/initialize**
```http
POST /api/initialize
Authorization: Bearer your-admin-api-key
Content-Type: application/json
```

Request:
```json
{
    "force_rebuild": false,
    "sources": ["react", "python", "fastapi"]
}
```

Response:
```json
{
    "status": "initialized",
    "documents_processed": 1547,
    "processing_time": 45.67,
    "message": "System initialized successfully"
}
```

## Request/Response Standards

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes (POST/PUT) | Must be `application/json` |
| `Authorization` | Admin only | `Bearer {api_key}` for admin endpoints |
| `User-Agent` | Recommended | Client identification |
| `Accept` | Optional | Response format preference |

### Response Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | Always `application/json; charset=utf-8` |
| `X-Response-Time` | Response processing time in milliseconds |
| `X-Rate-Limit-Remaining` | Remaining requests in current window |
| `X-Rate-Limit-Reset` | Rate limit window reset time |

### HTTP Status Codes

#### Success Codes
- **200 OK**: Successful GET, PUT, DELETE
- **201 Created**: Successful POST with resource creation
- **202 Accepted**: Async operation accepted
- **204 No Content**: Successful operation with no response body

#### Client Error Codes
- **400 Bad Request**: Invalid request format or missing required fields
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Valid auth but insufficient permissions
- **404 Not Found**: Resource doesn't exist
- **422 Unprocessable Entity**: Valid format but semantic errors
- **429 Too Many Requests**: Rate limit exceeded

#### Server Error Codes
- **500 Internal Server Error**: Unexpected server error
- **502 Bad Gateway**: Upstream service unavailable
- **503 Service Unavailable**: Service temporarily unavailable
- **504 Gateway Timeout**: Upstream service timeout

### Error Response Format

All error responses follow a consistent structure:

```json
{
    "error": "Human-readable error message",
    "code": "MACHINE_READABLE_CODE",
    "details": {
        "field": "problematic_field",
        "value": "invalid_value",
        "expected": "expected_format"
    },
    "request_id": "req_550e8400e29b41d4a716446655440000",
    "timestamp": "2024-08-10T19:48:00Z"
}
```

#### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request body or missing fields |
| `VALIDATION_ERROR` | 400 | Input validation failed |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `SERVICE_UNAVAILABLE` | 503 | OpenAI API or database unavailable |

### Response Envelope

Standard response structure for successful requests:

```json
{
    "data": {
        // Actual response data
    },
    "meta": {
        "response_time": 1.234,
        "request_id": "req_550e8400e29b41d4a716446655440000",
        "timestamp": "2024-08-10T19:48:00Z",
        "api_version": "1.0"
    }
}
```

## Implementation Patterns

### Request Validation

```python
from utils.validators import validate_json_structure

@main_bp.route('/api/chat', methods=['POST'])
def api_chat():
    # Validate JSON structure
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    
    # Validate required fields
    validation = validate_json_structure(data, required_fields=['query'])
    if not validation['valid']:
        return jsonify({
            'error': 'Invalid request format',
            'code': 'VALIDATION_ERROR',
            'details': validation['errors']
        }), 400
    
    # Validate query content
    query_validation = validate_query(data['query'])
    if not query_validation['valid']:
        return jsonify({
            'error': 'Invalid query',
            'code': 'VALIDATION_ERROR',
            'details': query_validation['errors']
        }), 400
```

### Response Building

```python
def build_api_response(data: Any, status_code: int = 200, 
                      response_time: float = None) -> tuple:
    """Build standardized API response."""
    response_data = {
        "data": data,
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request.headers.get('X-Request-ID', str(uuid.uuid4())),
            "api_version": "1.0"
        }
    }
    
    if response_time:
        response_data["meta"]["response_time"] = response_time
    
    return jsonify(response_data), status_code

# Usage
@main_bp.route('/api/stats')
def api_stats():
    start_time = time.time()
    
    stats = get_system_stats()
    response_time = time.time() - start_time
    
    return build_api_response(stats, response_time=response_time)
```

### Error Handling

```python
def handle_api_error(error: Exception, request_id: str = None) -> tuple:
    """Handle API errors consistently."""
    if isinstance(error, ValidationError):
        return jsonify({
            'error': error.message,
            'code': 'VALIDATION_ERROR',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat() + "Z"
        }), 400
    
    elif isinstance(error, openai.APIError):
        logger.error(f"OpenAI API error: {error}")
        return jsonify({
            'error': 'Service temporarily unavailable',
            'code': 'SERVICE_UNAVAILABLE',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat() + "Z"
        }), 503
    
    else:
        logger.exception(f"Unexpected error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat() + "Z"
        }), 500

# Global error handler
@app.errorhandler(Exception)
def handle_unexpected_error(error):
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    return handle_api_error(error, request_id)
```

## Rate Limiting

### Rate Limit Headers

Include rate limiting information in responses:

```python
@main_bp.after_request
def add_rate_limit_headers(response):
    """Add rate limiting headers to responses."""
    ip_address = request.remote_addr
    rate_limit_info = get_rate_limit_info(ip_address)
    
    response.headers.update({
        'X-RateLimit-Limit': Config.RATE_LIMIT_PER_MINUTE,
        'X-RateLimit-Remaining': rate_limit_info['remaining'],
        'X-RateLimit-Reset': rate_limit_info['reset_time'],
    })
    
    return response
```

### Rate Limit Error Response

```json
{
    "error": "Rate limit exceeded",
    "code": "RATE_LIMIT_EXCEEDED",
    "details": {
        "limit": 10,
        "window": "1 minute",
        "retry_after": 45
    },
    "timestamp": "2024-08-10T19:48:00Z"
}
```

## Authentication & Authorization

### API Key Authentication

```python
def require_admin_auth(f):
    """Decorator to require admin authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not validate_admin_key(auth_header):
            return jsonify({
                'error': 'Unauthorized access',
                'code': 'UNAUTHORIZED'
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function

@main_bp.route('/api/admin/initialize', methods=['POST'])
@require_admin_auth
def admin_initialize():
    # Admin-only functionality
    pass
```

### Session Management

```python
@main_bp.route('/api/chat', methods=['POST'])
def api_chat():
    # Ensure session exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    # Use session_id for conversation tracking
```

## Pagination

### Pagination Parameters

For endpoints returning lists:

```http
GET /api/conversations?page=1&limit=20&sort=created_at&order=desc
```

### Pagination Response

```json
{
    "data": [
        // Array of items
    ],
    "pagination": {
        "page": 1,
        "limit": 20,
        "total": 157,
        "pages": 8,
        "has_next": true,
        "has_prev": false
    }
}
```

### Implementation

```python
def paginate_query(query, page: int = 1, limit: int = 20):
    """Add pagination to SQLAlchemy query."""
    # Validate parameters
    page = max(1, page)
    limit = min(100, max(1, limit))  # Cap at 100 items
    
    # Execute paginated query
    paginated = query.paginate(
        page=page,
        per_page=limit,
        error_out=False
    )
    
    return {
        "data": [item.to_dict() for item in paginated.items],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": paginated.total,
            "pages": paginated.pages,
            "has_next": paginated.has_next,
            "has_prev": paginated.has_prev
        }
    }
```

## Content Negotiation

### Accept Headers

Support different response formats:

```python
@main_bp.route('/api/stats')
def api_stats():
    stats = get_system_stats()
    
    # Check Accept header
    if request.headers.get('Accept') == 'text/plain':
        return format_stats_as_text(stats), 200, {'Content-Type': 'text/plain'}
    else:
        return jsonify(stats)
```

### Content-Type Validation

```python
@main_bp.before_request
def validate_content_type():
    """Validate Content-Type for POST/PUT requests."""
    if request.method in ['POST', 'PUT']:
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
```

## Security Considerations

### CORS Configuration

```python
from flask_cors import CORS

def configure_cors(app):
    """Configure CORS based on environment."""
    if Config._is_production():
        # Restrictive CORS for production
        CORS(app, 
             origins=['https://yourdomain.com'],
             methods=['GET', 'POST'],
             allow_headers=['Content-Type', 'Authorization'])
    else:
        # Permissive CORS for development
        CORS(app, origins=['http://localhost:*'])
```

### Request Size Limits

```python
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB limit

@app.errorhandler(413)
def handle_request_too_large(error):
    return jsonify({
        'error': 'Request payload too large',
        'code': 'PAYLOAD_TOO_LARGE',
        'max_size': '1MB'
    }), 413
```

## API Testing

### Unit Tests

```python
def test_api_chat_valid_request():
    """Test valid chat API request."""
    with app.test_client() as client:
        response = client.post('/api/chat', 
                              json={'query': 'What is Python?'},
                              content_type='application/json')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'response' in data
        assert 'sources' in data
        assert 'session_id' in data

def test_api_chat_invalid_request():
    """Test invalid chat API request."""
    with app.test_client() as client:
        response = client.post('/api/chat', 
                              json={},  # Missing query
                              content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert data['code'] == 'VALIDATION_ERROR'
```

### Integration Tests

```python
def test_chat_workflow_integration():
    """Test complete chat workflow."""
    with app.test_client() as client:
        # Send chat request
        response = client.post('/api/chat', 
                              json={'query': 'What is React?'})
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Verify response structure
        assert isinstance(data['response'], str)
        assert isinstance(data['sources'], list)
        assert len(data['sources']) > 0
        assert data['response_time'] > 0
        
        # Verify conversation is stored
        session_id = data['session_id']
        conversation = Conversation.query.filter_by(session_id=session_id).first()
        assert conversation is not None
        assert conversation.user_query == 'What is React?'
```

## Performance Optimization

### Response Caching

```python
from cache_manager import CacheManager

cache = CacheManager()

@main_bp.route('/api/stats')
def api_stats():
    cache_key = "api_stats"
    cached_stats = cache.get(cache_key)
    
    if cached_stats:
        return jsonify(cached_stats)
    
    stats = compute_expensive_stats()
    cache.set(cache_key, stats, ttl=300)  # Cache for 5 minutes
    
    return jsonify(stats)
```

### Async Processing

For long-running operations:

```python
@main_bp.route('/api/documents/reindex', methods=['POST'])
@require_admin_auth
def reindex_documents():
    """Start document reindexing process asynchronously."""
    task_id = str(uuid.uuid4())
    
    # Start background task
    from threading import Thread
    thread = Thread(target=background_reindex, args=(task_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'accepted',
        'task_id': task_id,
        'message': 'Reindexing started',
        'check_status_url': f'/api/tasks/{task_id}'
    }), 202
```

## API Monitoring

### Response Time Tracking

```python
@main_bp.before_request
def start_timer():
    """Start request timer."""
    request.start_time = time.time()

@main_bp.after_request
def add_response_time(response):
    """Add response time header."""
    if hasattr(request, 'start_time'):
        response_time = (time.time() - request.start_time) * 1000
        response.headers['X-Response-Time'] = f"{response_time:.2f}ms"
        
        # Log slow responses
        if response_time > 5000:  # 5 seconds
            logger.warning(f"Slow API response: {request.endpoint} took {response_time:.2f}ms")
    
    return response
```

### Request Logging

```python
@main_bp.before_request
def log_request():
    """Log API requests."""
    # Don't log sensitive data
    safe_args = {k: v for k, v in request.args.items() 
                 if k not in ['api_key', 'token', 'password']}
    
    logger.info(f"API Request: {request.method} {request.path} "
                f"from {request.remote_addr} args={safe_args}")
```

## API Documentation Generation

### OpenAPI/Swagger Integration

```python
from flask import Flask
from flask_restx import Api, Resource, fields

api = Api(app, doc='/api/docs/', title='DocRag API', 
          description='AI-powered documentation assistant API')

# Define models
chat_request_model = api.model('ChatRequest', {
    'query': fields.String(required=True, description='User question'),
    'context': fields.String(description='Additional context')
})

chat_response_model = api.model('ChatResponse', {
    'response': fields.String(description='AI generated response'),
    'sources': fields.List(fields.String, description='Source URLs'),
    'response_time': fields.Float(description='Processing time in seconds'),
    'session_id': fields.String(description='Session identifier')
})

# Use models in endpoints
@api.route('/chat')
class ChatAPI(Resource):
    @api.expect(chat_request_model)
    @api.marshal_with(chat_response_model)
    def post(self):
        """Process chat query and return AI response."""
        # Implementation
        pass
```

---

*This API design ensures consistency, security, and maintainability while providing a great developer experience. Keep the API documentation up-to-date as endpoints evolve.*