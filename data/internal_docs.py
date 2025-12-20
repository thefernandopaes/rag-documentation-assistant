"""
Internal Documentation for RAG Documentation Assistant

This file contains comprehensive documentation about the RAG Documentation Assistant
system itself. This documentation is indexed in ChromaDB so the AI can answer
questions about its own capabilities, endpoints, and features.
"""

INTERNAL_DOCUMENTATION = {
    "system_overview": {
        "title": "RAG Documentation Assistant - System Overview",
        "content": """
# RAG Documentation Assistant

## What is this system?

The RAG Documentation Assistant (DocRag) is an AI-powered documentation assistant specialized in API documentation. It uses **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers about APIs.

## Key Features

### 1. API Documentation Discovery
- Automatically discovers and processes API documentation from various sources
- Supports OpenAPI/Swagger, REST APIs, GraphQL, and Postman collections
- Indexes documentation into a searchable vector database (ChromaDB)

### 2. Intelligent Code Generation
- Generates code examples in multiple languages:
  - cURL (command-line)
  - Python (requests, httpx)
  - JavaScript/Node.js (fetch, axios)
  - PHP, Ruby, Go, Java
- Automatically adapts examples to the specific API endpoint

### 3. Context-Aware Responses
- Maintains conversation history for follow-up questions
- Provides relevant code examples and endpoint information
- Suggests related concepts and questions

### 4. Session-Based Authentication
- Uses cookie-based sessions (no OAuth2 or API keys for basic use)
- Each user gets a unique session ID for conversation tracking
- Sessions persist for 30 days

## Architecture

### Technology Stack
- **Backend**: FastAPI (async Python framework)
- **Database**: SQLite with async driver (aiosqlite)
- **Vector Database**: ChromaDB with OpenAI embeddings
- **AI Model**: OpenAI GPT-4 with async client
- **Frontend**: Bootstrap 5 with dark theme
- **Syntax Highlighting**: Prism.js with multiple language support

### Performance
- Async/await throughout for 2-4x faster response times
- Caching system with 3600s TTL for frequently asked questions
- Rate limiting: 10 requests/minute per session

## Use Cases
1. Learning how to use a new API
2. Generating quick code examples for API endpoints
3. Understanding authentication flows
4. Troubleshooting API errors
5. Discovering related API endpoints
        """,
        "type": "internal",
        "url": "internal://system-overview",
        "doc_type": "internal_docs",
        "version": "2.0.0"
    },

    "authentication": {
        "title": "RAG Documentation Assistant - Authentication",
        "content": """
# Authentication with DocRag API

## For Regular Users (Chat Interface)

### Automatic Session-Based Authentication

This API uses **session-based authentication** with cookies. **No API keys or OAuth2 tokens are required** for basic usage.

#### How It Works

1. **Automatic Session Creation**
   - When you make your first request to `/api/chat`, a session cookie is automatically created
   - The session ID is a UUID v4 stored in your browser
   - No manual authentication needed

2. **Session Cookie Details**
   - Cookie name: `session_id`
   - Max age: 30 days (2,592,000 seconds)
   - HttpOnly: Yes (secure)
   - SameSite: Lax

3. **Conversation Tracking**
   - Your session ID links all your conversations
   - Conversation history is maintained for context
   - Up to 5 recent exchanges are kept in memory

### Example Usage

#### Making a Chat Request
```bash
curl -X POST http://localhost:8000/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{"query": "How to authenticate with this API?"}'
```

The server will automatically create a session cookie in the response.

#### Getting Your History
```bash
curl http://localhost:8000/api/history \\
  -H "Cookie: session_id=your-session-id-here"
```

## For Admins (Protected Endpoints)

### Admin API Key Authentication

Some endpoints like `/api/initialize` require admin authentication.

#### Setting Up Admin Key
Set the `ADMIN_API_KEY` environment variable:
```bash
export ADMIN_API_KEY=your-secure-admin-key-here
```

#### Using Admin Endpoints
```bash
curl -X POST http://localhost:8000/api/initialize \\
  -H "X-Admin-Key: your-secure-admin-key-here" \\
  -H "Content-Type: application/json"
```

## Security Features

1. **HTTPS Enforcement** (production only)
2. **CORS Configuration** with allowed domains
3. **Request Size Limits**: 16KB max per request
4. **Rate Limiting**: 10 requests/minute per session
5. **XSS Protection Headers**
6. **SQL Injection Prevention** via Pydantic validation
7. **Session Expiration** after 30 days of inactivity

## No OAuth2 or JWT

**Important**: This API does NOT use OAuth2, JWT tokens, or bearer authentication for regular users. All authentication is session-based with automatic cookie management.
        """,
        "type": "internal",
        "url": "internal://authentication",
        "doc_type": "internal_docs",
        "version": "2.0.0"
    },

    "api_endpoints": {
        "title": "RAG Documentation Assistant - API Endpoints",
        "content": """
# API Endpoints Reference

## Chat Endpoints

### POST /api/chat
Generate AI responses using RAG.

**Authentication**: Session cookie (automatic)
**Rate Limit**: 10 requests/minute

**Request**:
```json
{
  "query": "How to create a FastAPI endpoint?",
  "session_id": "optional-uuid"
}
```

**Response**:
```json
{
  "response": "To create a FastAPI endpoint...",
  "sources": [
    {
      "title": "FastAPI Documentation",
      "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
      "type": "api",
      "relevance": 0.95
    }
  ],
  "examples": [
    {
      "language": "python",
      "code": "@app.get('/items/{item_id}')\\ndef read_item(item_id: int):\\n    return {'item_id': item_id}",
      "explanation": "Basic GET endpoint example"
    }
  ],
  "response_time": 2.5,
  "cached": false,
  "related_concepts": ["routing", "path parameters", "request validation"]
}
```

### POST /api/feedback
Submit feedback for a conversation.

**Authentication**: Session cookie
**Rate Limit**: 5 requests/minute

**Request**:
```json
{
  "conversation_id": "uuid-of-conversation",
  "feedback": 1  // 1 for positive, -1 for negative
}
```

**Response**:
```json
{
  "message": "Feedback recorded successfully"
}
```

### GET /api/history
Retrieve conversation history for your session.

**Authentication**: Session cookie
**Query Parameters**:
- `limit` (optional): Max conversations to return (default: 20)

**Response**:
```json
[
  {
    "id": "conversation-uuid",
    "query": "How to use FastAPI?",
    "response": "FastAPI is...",
    "sources": [...],
    "created_at": "2025-12-19T10:30:00",
    "response_time": 2.3,
    "feedback": 1
  }
]
```

## System Endpoints

### GET /api/stats
Get system statistics and health information.

**Authentication**: None

**Response**:
```json
{
  "documents": {
    "document_count": 62,
    "collection_name": "api_documentation"
  },
  "conversations": {
    "total": 150,
    "avg_response_time": 2.8,
    "positive_feedback": 120,
    "negative_feedback": 10
  },
  "cache": {
    "total_entries": 45,
    "hit_rate": 0.65
  },
  "system": {
    "is_production": false,
    "version": "2.0.0",
    "framework": "fastapi",
    "async": true
  }
}
```

### GET /health
Health check endpoint for monitoring.

**Authentication**: None

**Response**:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "framework": "fastapi",
  "async": true
}
```

## Admin Endpoints

### POST /api/initialize
Initialize the RAG system with documentation sources.

**Authentication**: Admin API key (X-Admin-Key header)
**Production Only**: Requires admin key in production

**Request**:
```json
{
  "force": false  // Set to true to re-initialize
}
```

**Response**:
```json
{
  "message": "System initialized successfully with 62 documents",
  "status": "initialized",
  "document_count": 62
}
```

## Frontend Routes

### GET /
Landing page with system overview and features.

### GET /chat
Interactive chat interface.

### GET /docs
Auto-generated OpenAPI/Swagger documentation.

### GET /redoc
Alternative API documentation (ReDoc).

## Error Responses

All endpoints return consistent error format:
```json
{
  "error": "Error description",
  "status_code": 400,
  "path": "/api/chat",
  "details": [...]  // Only for validation errors
}
```

## Common Status Codes
- **200 OK**: Success
- **400 Bad Request**: Invalid input
- **401 Unauthorized**: Missing/invalid admin key
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
        """,
        "type": "internal",
        "url": "internal://api-endpoints",
        "doc_type": "internal_docs",
        "version": "2.0.0"
    },

    "code_examples": {
        "title": "RAG Documentation Assistant - Code Examples",
        "content": """
# Code Example Generation

The RAG Documentation Assistant can automatically generate code examples in multiple languages for API endpoints.

## Supported Languages

1. **cURL** - Command-line HTTP requests
2. **Python** - Using requests or httpx libraries
3. **JavaScript/Node.js** - Using fetch or axios
4. **PHP** - Using cURL or Guzzle
5. **Ruby** - Using net/http or HTTParty
6. **Go** - Using net/http package
7. **Java** - Using HttpClient or OkHttp

## Example: Chat Request in Different Languages

### cURL
```bash
curl -X POST http://localhost:8000/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{"query": "How to authenticate?"}'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    json={"query": "How to authenticate?"}
)
data = response.json()
print(data["response"])
```

### JavaScript (Fetch)
```javascript
fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'How to authenticate?'})
})
.then(res => res.json())
.then(data => console.log(data.response));
```

### JavaScript (Node.js with axios)
```javascript
const axios = require('axios');

axios.post('http://localhost:8000/api/chat', {
  query: 'How to authenticate?'
})
.then(response => {
  console.log(response.data.response);
});
```

## How Code Generation Works

1. **Context Analysis**: The AI analyzes your query to understand:
   - The endpoint you're asking about
   - Required parameters and headers
   - Authentication method needed

2. **Template Selection**: Selects appropriate code templates for each language

3. **Parameter Injection**: Fills in:
   - Endpoint URLs
   - Required headers
   - Request body structure
   - Authentication tokens (if applicable)

4. **Error Handling**: Includes basic error handling where appropriate

5. **Comments**: Adds explanatory comments for clarity

## Requesting Specific Languages

You can request examples in specific languages:
- "Show me a cURL example for..."
- "How to do this in Python?"
- "Give me JavaScript code for..."
- "Show me the Go implementation..."

## Features

### Automatic Adaptation
- Examples adapt to the specific API being discussed
- Authentication headers are included when needed
- Request/response formats match the API specification

### Multiple Variants
- For languages with multiple HTTP libraries, we provide the most popular option
- Example: Python uses `requests` (most popular), but can also show `httpx` for async

### Working Examples
- All generated code is functional and can be copied directly
- Includes necessary imports and dependencies
- Shows expected output format
        """,
        "type": "internal",
        "url": "internal://code-examples",
        "doc_type": "internal_docs",
        "version": "2.0.0"
    },

    "rate_limiting": {
        "title": "RAG Documentation Assistant - Rate Limiting",
        "content": """
# Rate Limiting

The DocRag API implements rate limiting to ensure fair usage and system stability.

## Limits

### Chat Endpoint (/api/chat)
- **Limit**: 10 requests per minute
- **Window**: 60 seconds rolling window
- **Scope**: Per session ID

### Feedback Endpoint (/api/feedback)
- **Limit**: 5 requests per minute
- **Window**: 60 seconds rolling window
- **Scope**: Per session ID

### Other Endpoints
- **No rate limiting** on read-only endpoints like /api/stats, /health

## How It Works

1. **Session Tracking**: Rate limits are tracked per session ID (cookie-based)
2. **Rolling Window**: Limits are calculated over a 60-second rolling window
3. **Reset**: Limits reset automatically after the time window expires

## Rate Limit Headers

When rate limited, you'll receive:
- **Status Code**: 429 Too Many Requests
- **Retry-After** header: Seconds until you can try again

## Example Rate Limit Response

```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Try again in 45 seconds.",
  "retry_after": 45,
  "status_code": 429
}
```

## Best Practices

1. **Implement Exponential Backoff**: If you receive a 429, wait before retrying
2. **Cache Responses**: The system already caches responses internally
3. **Batch Requests**: Combine related questions when possible
4. **Use Conversation History**: The system maintains context, so you don't need to repeat information

## For Higher Limits

Rate limits are configurable via environment variables. Contact system admin for increased limits if needed.

**Environment Variables**:
- `RATE_LIMIT_CHAT`: Requests per minute for /api/chat (default: 10)
- `RATE_LIMIT_FEEDBACK`: Requests per minute for /api/feedback (default: 5)
        """,
        "type": "internal",
        "url": "internal://rate-limiting",
        "doc_type": "internal_docs",
        "version": "2.0.0"
    },

    "features": {
        "title": "RAG Documentation Assistant - Features & Capabilities",
        "content": """
# Features & Capabilities

## What Can DocRag Do?

### 1. Answer API Documentation Questions
Ask natural language questions about APIs:
- "How do I authenticate with this API?"
- "Show me all available endpoints"
- "What parameters does the POST /users endpoint accept?"
- "How do I handle rate limiting?"

### 2. Generate Code Examples
Automatically generate working code in multiple languages:
- cURL commands for testing
- Python code using requests
- JavaScript/TypeScript with fetch or axios
- And 5 more languages

### 3. Explain API Concepts
Get clear explanations of:
- REST API principles
- HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Status codes (200, 404, 500, etc.)
- Authentication types (API keys, OAuth2, JWT)
- Request/response formats (JSON, XML)

### 4. Discover Related Information
The system suggests:
- Related API endpoints
- Follow-up questions to ask
- Alternative approaches
- Best practices

### 5. Maintain Conversation Context
- Remembers previous questions in your session
- Allows follow-up questions without repeating context
- Keeps last 5 exchanges in memory

### 6. Source Citations
Every answer includes:
- Links to official documentation
- Relevance scores for sources
- Document types (OpenAPI, REST, GraphQL, etc.)

## Advanced Features

### API-Specific Responses
When answering questions about APIs, responses include:

#### Endpoints Information
- HTTP method (GET, POST, etc.)
- Full endpoint path
- Description of what it does

#### Authentication Details
- Type of authentication (API key, OAuth2, etc.)
- How to include credentials
- Token endpoints (if OAuth2)

#### Parameters
- Required vs optional parameters
- Parameter types (string, integer, etc.)
- Descriptions and constraints

#### Response Format
- Expected response structure
- Field descriptions
- Example responses

#### Error Codes
- Common error codes for the endpoint
- What each error means
- How to handle them

### Caching
- Frequently asked questions are cached
- Cache duration: 1 hour
- Indicated in response with `"cached": true`
- Significantly faster responses for cached queries

### Performance
- Average response time: 2-3 seconds
- Cached responses: < 0.5 seconds
- Async architecture for concurrent requests
- Handles 10-20 simultaneous users

## What DocRag Cannot Do

### Limitations
1. **Cannot access private APIs** - Only works with publicly accessible documentation
2. **Cannot execute API calls** - Only generates example code, doesn't run it
3. **No real-time API data** - Information based on indexed documentation
4. **Cannot modify APIs** - Read-only access to documentation

### Out of Scope
- Writing production-ready applications
- Debugging your API code
- Hosting or deploying APIs
- API key generation
- Direct database access

## Best Use Cases

### Ideal For:
✅ Learning how to use a new API
✅ Quick code example generation
✅ Understanding API concepts
✅ Exploring API capabilities
✅ Getting started with API integration

### Not Ideal For:
❌ Complex multi-step workflows
❌ Production code generation
❌ Real-time API monitoring
❌ API performance testing
❌ Debugging live API issues
        """,
        "type": "internal",
        "url": "internal://features",
        "doc_type": "internal_docs",
        "version": "2.0.0"
    }
}


def get_all_internal_docs():
    """
    Get all internal documentation as a list of dictionaries.

    Returns:
        List[dict]: List of internal documentation chunks
    """
    docs = []
    for key, doc in INTERNAL_DOCUMENTATION.items():
        docs.append({
            'title': doc['title'],
            'content': doc['content'],
            'source_url': doc['url'],
            'doc_type': doc.get('doc_type', 'internal_docs'),
            'version': doc.get('version', '2.0.0'),
            'type': doc.get('type', 'internal')
        })
    return docs


def get_internal_doc(key: str):
    """
    Get a specific internal documentation by key.

    Args:
        key: The documentation key (e.g., 'authentication', 'api_endpoints')

    Returns:
        dict: The documentation or None if not found
    """
    return INTERNAL_DOCUMENTATION.get(key)
