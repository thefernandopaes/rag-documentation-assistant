# System Architecture

This document describes the overall system architecture, components, and data flow of the DocRag application.

## Overview

DocRag is a production-ready RAG (Retrieval-Augmented Generation) system that provides AI-powered assistance for technical documentation. The system follows a modular, microservice-oriented architecture with clear separation of concerns.

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│   Web Client    │◄──►│   Flask App      │◄──►│   PostgreSQL    │
│   (Frontend)    │    │   (Backend)      │    │   (Metadata)    │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │                  │    │                 │
                       │   RAG Engine     │◄──►│   ChromaDB      │
                       │                  │    │   (Vectors)     │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │                  │    │                 │
                       │ Document Proc.   │◄──►│   OpenAI API    │
                       │                  │    │   (Embeddings)  │
                       └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. Flask Web Application (`app.py`)

**Responsibility**: Application factory, configuration, and initialization

**Key Features**:
- Application factory pattern
- Database initialization
- Blueprint registration
- Production security configuration (ProxyFix, secure cookies)
- RAG engine initialization

**Dependencies**:
- Flask-SQLAlchemy for database ORM
- Werkzeug ProxyFix for production deployment
- Configuration management

```python
def create_app():
    app = Flask(__name__)
    # Security configuration for production
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
    # Database and extensions setup
    db.init_app(app)
    # RAG engine initialization
    app.rag_engine = RAGEngine()
    return app
```

### 2. Configuration Management (`config.py`)

**Responsibility**: Centralized configuration with environment-based settings

**Key Features**:
- Environment variable loading with defaults
- Configuration validation
- Database URI normalization
- Environment detection (production/development)
- Writable directory validation

**Configuration Groups**:
- OpenAI settings (API key, model, timeouts)
- Database settings (URI, pooling, SSL)
- RAG settings (chunk size, overlap, temperature)
- Crawler settings (delays, timeouts, limits)
- Cache and rate limiting settings

### 3. Database Models (`models.py`)

**Responsibility**: Data persistence and relationships

**Models**:

#### Conversation
- Stores chat interactions and user queries
- Tracks response time and feedback
- Links to session management

#### DocumentChunk
- Represents processed document segments
- Contains content hash for idempotency
- Maps to ChromaDB embedding IDs
- Tracks source metadata

#### RateLimit
- IP-based request throttling
- Request counting and reset times
- Abuse prevention

```python
class DocumentChunk(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    source_url = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    content_hash = db.Column(db.String(64))  # SHA-256 for idempotency
    embedding_id = db.Column(db.String(100))  # ChromaDB reference
```

### 4. RAG Engine (`rag_engine.py`)

**Responsibility**: Core RAG functionality - retrieval and generation

**Key Components**:
- **Vector Store Management**: ChromaDB integration with cosine similarity
- **Embeddings**: OpenAI text-embedding-3-small integration
- **Text Chunking**: LangChain RecursiveCharacterTextSplitter
- **Generation**: GPT-4o for response synthesis
- **Caching**: Query result caching for performance

**Data Flow**:
1. **Indexing**: Document → Chunks → Embeddings → ChromaDB
2. **Retrieval**: Query → Embedding → Similarity Search → Top-k chunks
3. **Generation**: Context + Query → GPT-4o → Structured response

```python
class RAGEngine:
    def query(self, query: str, conversation_history: List[Dict]) -> Dict:
        # 1. Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # 2. Retrieve similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # 3. Build context and generate response
        context = self._build_context(results, conversation_history)
        response = self._generate_response(query, context)
        
        return response
```

### 5. Document Processor (`document_processor.py`)

**Responsibility**: Content crawling, extraction, and preprocessing

**Key Features**:
- **Web Crawling**: Requests + BeautifulSoup for HTML parsing
- **Content Extraction**: Trafilatura for clean text extraction
- **URL Management**: Sitemap parsing and link discovery
- **Rate Limiting**: Configurable delays between requests
- **Sample Data**: Development mode with predefined content

**Processing Pipeline**:
1. URL discovery (sitemap or link crawling)
2. Content fetching with timeout and retry logic
3. HTML to text extraction
4. Content cleaning and normalization
5. Metadata extraction (title, URL, type)

### 6. Routes (`routes.py`)

**Responsibility**: HTTP API endpoints and request handling

**Endpoints**:
- `GET /`: Home page
- `GET /chat`: Chat interface
- `POST /api/chat`: Main query endpoint
- `GET /api/stats`: System statistics
- `POST /api/initialize`: Admin initialization
- `GET /healthz`: Health check

**Features**:
- Rate limiting decorator
- Session management
- Input validation and sanitization
- Error handling and logging
- Response time tracking

### 7. Cache Manager (`cache_manager.py`)

**Responsibility**: File-based caching for performance optimization

**Features**:
- TTL-based cache expiration
- File-based storage for persistence
- JSON serialization for complex objects
- Cache key hashing for safe filenames
- Automatic cleanup of expired entries

### 8. Rate Limiter (`rate_limiter.py`)

**Responsibility**: Request throttling and abuse prevention

**Features**:
- IP-based rate limiting
- Sliding window implementation
- Database-backed counters
- Configurable limits per minute
- Automatic reset logic

### 9. Input Validation (`utils/validators.py`)

**Responsibility**: Security-focused input validation

**Validation Types**:
- Query validation (length, content, XSS prevention)
- API key format validation
- Email and username validation
- File upload validation
- JSON structure validation
- SQL injection prevention

## Data Flow

### Query Processing Flow

```
1. User Query
   ↓
2. Input Validation & Sanitization
   ↓
3. Rate Limit Check
   ↓
4. Session Management
   ↓
5. Cache Lookup
   ↓ (cache miss)
6. Query Embedding Generation (OpenAI)
   ↓
7. Vector Similarity Search (ChromaDB)
   ↓
8. Context Building + History
   ↓
9. Response Generation (GPT-4o)
   ↓
10. Response Caching
    ↓
11. Database Logging
    ↓
12. JSON Response to Client
```

### Document Ingestion Flow

```
1. Admin Trigger (scripts/ingest.py)
   ↓
2. Documentation Source Configuration
   ↓
3. URL Discovery (sitemap/crawling)
   ↓
4. Content Fetching (rate-limited)
   ↓
5. HTML → Text Extraction
   ↓
6. Text Chunking (LangChain)
   ↓
7. Content Hash Generation (SHA-256)
   ↓
8. Duplicate Check (idempotency)
   ↓
9. Embedding Generation (OpenAI)
   ↓
10. Vector Storage (ChromaDB)
    ↓
11. Metadata Storage (PostgreSQL)
```

## Storage Architecture

### PostgreSQL Database

**Purpose**: Metadata, chat history, and system state

**Tables**:
- `conversation`: User interactions and feedback
- `document_chunk`: Document metadata and hashes
- `rate_limit`: IP-based throttling data

**Indexes**:
- `conversation.session_id` for chat history
- `document_chunk.source_url` for deduplication
- `rate_limit.ip_address` for quick lookups

### ChromaDB Vector Store

**Purpose**: Embedding storage and similarity search

**Configuration**:
- **Distance Metric**: Cosine similarity
- **Collection**: Single collection for all document types
- **Persistence**: File-based storage
- **Metadata**: Document type, URL, chunk index

### File System Cache

**Purpose**: Response caching and performance optimization

**Structure**:
```
cache/
├── <hash1>.cache  # Cached query responses
├── <hash2>.cache  # Cached embeddings
└── <hash3>.cache  # Cached API responses
```

## Security Architecture

### Input Security
- XSS prevention through HTML escaping
- SQL injection prevention via parameterized queries
- Input length and format validation
- Malicious content pattern detection

### API Security
- Rate limiting per IP address
- Admin API key protection
- Session management with secure cookies
- HTTPS enforcement in production

### Data Security
- Environment variable configuration
- No hardcoded secrets
- Database SSL connections
- Secure session cookie settings

## Performance Architecture

### Caching Strategy
- **L1 Cache**: In-memory response caching
- **L2 Cache**: File-based persistent cache
- **L3 Cache**: Database query result caching

### Database Optimization
- Connection pooling (configurable size)
- Query optimization with proper indexes
- Connection recycling and health checks

### Vector Search Optimization
- Efficient similarity search with ChromaDB
- Embedding reuse for identical queries
- Result limiting to top-k most relevant

## Scalability Considerations

### Horizontal Scaling
- Stateless application design
- External session storage capability
- Database connection pooling
- Load balancer compatibility

### Vertical Scaling
- Configurable worker processes (Gunicorn)
- Adjustable database pool sizes
- Memory-efficient text processing
- Streaming response capability

## Deployment Architecture

### Production Environment
```
[Load Balancer] → [Gunicorn Workers] → [Flask App]
                                    ↓
                            [PostgreSQL Database]
                                    ↓
                            [Persistent Volume (ChromaDB)]
```

### Container Considerations
- Persistent volumes for ChromaDB and cache
- Environment variable configuration
- Health check endpoints
- Graceful shutdown handling

## Monitoring and Observability

### Health Checks
- `/healthz`: Database connectivity and vector count
- `/api/stats`: System statistics and performance metrics
- Database connection status
- ChromaDB collection status

### Logging
- Structured logging with context
- Error tracking and alerting
- Performance metrics logging
- Request/response logging

### Metrics
- Response time tracking
- Cache hit/miss rates
- Rate limiting statistics
- Database query performance

---

*This architecture supports the current requirements while providing flexibility for future enhancements and scaling needs.*