# Performance Guidelines

This document outlines optimization strategies, performance best practices, and monitoring approaches for the DocRag application.

## Performance Targets

### Response Time Targets

| Operation | Target | Maximum | Notes |
|-----------|--------|---------|-------|
| Home page load | <200ms | 500ms | Static content |
| Chat API request | <2s | 5s | Including RAG processing |
| Health check | <50ms | 100ms | Database ping only |
| Statistics API | <100ms | 300ms | Cached data preferred |
| Document ingestion | N/A | 30s/doc | Batch operation |

### Resource Usage Targets

| Resource | Target | Maximum | Monitoring |
|----------|--------|---------|------------|
| Memory per worker | <300MB | 500MB | Process monitoring |
| Database connections | <5 per worker | 10 per worker | Pool monitoring |
| Disk usage (cache) | <100MB | 500MB | Log rotation |
| ChromaDB storage | <1GB | 5GB | Collection size |

## Caching Strategy

### Multi-Level Caching

The application implements a three-tier caching strategy:

```python
# L1: In-memory caching (fastest)
@lru_cache(maxsize=128)
def get_frequent_data(key: str):
    return expensive_computation(key)

# L2: File-based caching (persistent)
cache_manager = CacheManager()
result = cache_manager.get(cache_key)
if not result:
    result = compute_result()
    cache_manager.set(cache_key, result, ttl=3600)

# L3: Database query caching
@cached_query(ttl=300)
def get_conversation_history(session_id: str):
    return Conversation.query.filter_by(session_id=session_id).all()
```

### Cache Key Strategies

```python
def generate_cache_key(query: str, context: Dict = None) -> str:
    """Generate efficient cache keys for different operations."""
    
    # Normalize query for better cache hits
    normalized_query = re.sub(r'\s+', ' ', query.lower().strip())
    
    # Include relevant context
    cache_components = [normalized_query]
    
    if context:
        # Include only relevant context elements
        relevant_context = {
            'doc_type': context.get('doc_type'),
            'user_level': context.get('user_level', 'intermediate')
        }
        cache_components.append(json.dumps(relevant_context, sort_keys=True))
    
    # Generate hash
    cache_string = '|'.join(cache_components)
    return hashlib.sha256(cache_string.encode()).hexdigest()[:16]
```

### Cache Invalidation

```python
class SmartCacheManager:
    def __init__(self):
        self.cache = CacheManager()
        self.cache_tags = {}  # Track cache dependencies
    
    def set_with_tags(self, key: str, value: Any, tags: List[str], ttl: int):
        """Set cache value with dependency tags."""
        self.cache.set(key, value, ttl)
        
        # Track tags
        for tag in tags:
            if tag not in self.cache_tags:
                self.cache_tags[tag] = set()
            self.cache_tags[tag].add(key)
    
    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with specific tag."""
        if tag in self.cache_tags:
            for cache_key in self.cache_tags[tag]:
                self.cache.delete(cache_key)
            del self.cache_tags[tag]
            logger.info(f"Invalidated cache entries with tag: {tag}")

# Usage
cache_manager = SmartCacheManager()

# Cache with tags
cache_manager.set_with_tags(
    key="query:12345",
    value=response_data,
    tags=["react_docs", "api_responses"],
    ttl=3600
)

# Invalidate when React docs are updated
cache_manager.invalidate_by_tag("react_docs")
```

## Database Performance

### Query Optimization

#### Index Strategy

```sql
-- Essential indexes for performance
CREATE INDEX CONCURRENTLY idx_conversation_session_created 
ON conversation(session_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_document_chunk_type_hash 
ON document_chunk(doc_type, content_hash);

CREATE INDEX CONCURRENTLY idx_rate_limit_ip_reset 
ON rate_limit(ip_address, reset_time) 
WHERE reset_time > NOW();

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_conversation_feedback_recent 
ON conversation(feedback, created_at) 
WHERE feedback IS NOT NULL AND created_at > NOW() - INTERVAL '30 days';
```

#### Query Performance Monitoring

```python
import time
from functools import wraps

def monitor_db_performance(operation_name: str):
    """Decorator to monitor database operation performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log slow queries
                if duration > 1.0:  # Queries slower than 1 second
                    logger.warning(f"Slow DB operation: {operation_name} took {duration:.3f}s")
                
                # Track performance metrics
                track_db_performance(operation_name, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"DB operation failed: {operation_name} after {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator

@monitor_db_performance("get_conversation_history")
def get_conversation_history(session_id: str, limit: int = 10):
    """Get conversation history with performance monitoring."""
    return Conversation.query\
        .filter_by(session_id=session_id)\
        .order_by(Conversation.created_at.desc())\
        .limit(limit)\
        .all()
```

#### Connection Pool Optimization

```python
def optimize_db_pool():
    """Optimize database connection pool based on usage patterns."""
    
    # Monitor pool usage
    pool_stats = get_db_pool_status()
    
    # Adjust pool size based on utilization
    if pool_stats['utilization'] > 0.8:  # 80% utilization
        logger.warning("High database pool utilization detected")
        
        # Recommendations for scaling
        current_size = Config.DB_POOL_SIZE
        recommended_size = min(current_size * 2, 20)  # Cap at 20
        
        logger.info(f"Consider increasing DB_POOL_SIZE from {current_size} to {recommended_size}")
    
    return pool_stats
```

### Batch Operations

```python
def batch_insert_conversations(conversations: List[Dict]) -> None:
    """Efficiently insert multiple conversations."""
    
    # Use bulk operations for better performance
    conversation_objects = [
        Conversation(**conv_data) for conv_data in conversations
    ]
    
    try:
        # Bulk insert
        db.session.bulk_save_objects(conversation_objects)
        db.session.commit()
        
        logger.info(f"Bulk inserted {len(conversations)} conversations")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Bulk insert failed: {e}")
        
        # Fallback: Individual inserts with error handling
        successful_inserts = 0
        for conv_data in conversations:
            try:
                conv = Conversation(**conv_data)
                db.session.add(conv)
                db.session.commit()
                successful_inserts += 1
            except Exception as individual_error:
                db.session.rollback()
                logger.warning(f"Individual insert failed: {individual_error}")
        
        logger.info(f"Fallback: {successful_inserts}/{len(conversations)} conversations inserted")
```

## RAG Performance Optimization

### Embedding Optimization

#### Batch Embedding Generation

```python
async def generate_embeddings_batch(self, texts: List[str], 
                                  batch_size: int = 20) -> List[List[float]]:
    """Generate embeddings in batches for better throughput."""
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        try:
            # Batch API call to OpenAI
            response = await self.openai_client.embeddings.acreate(
                model="text-embedding-3-small",
                input=batch_texts
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting between batches
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Batch embedding failed for batch {i//batch_size + 1}: {e}")
            
            # Fallback: Process individually
            for text in batch_texts:
                try:
                    embedding = await self._generate_single_embedding(text)
                    all_embeddings.append(embedding)
                except Exception as individual_error:
                    logger.warning(f"Individual embedding failed: {individual_error}")
                    all_embeddings.append([0.0] * 1536)  # Zero vector fallback
    
    return all_embeddings
```

#### Embedding Caching

```python
class EmbeddingCache:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.hit_count = 0
        self.miss_count = 0
    
    def get_or_generate(self, text: str) -> List[float]:
        """Get embedding from cache or generate new one."""
        
        # Generate cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"embedding:{text_hash}"
        
        # Check cache first
        cached_embedding = self.cache.get(cache_key)
        if cached_embedding:
            self.hit_count += 1
            return cached_embedding
        
        # Generate new embedding
        self.miss_count += 1
        embedding = self._generate_embedding(text)
        
        # Cache for 24 hours (embeddings are stable)
        self.cache.set(cache_key, embedding, ttl=86400)
        
        return embedding
    
    def get_cache_stats(self) -> Dict:
        """Get embedding cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
```

### Vector Search Optimization

#### Optimized Retrieval

```python
def retrieve_with_optimization(self, query: str, 
                              doc_type: str = None,
                              n_results: int = 5) -> List[Dict]:
    """Optimized document retrieval with filtering."""
    
    # Generate query embedding (with caching)
    query_embedding = self.embedding_cache.get_or_generate(query)
    
    # Build where clause for filtering
    where_clause = {}
    if doc_type:
        where_clause["doc_type"] = doc_type
    
    # Perform search with metadata filtering
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2,  # Get more results for post-filtering
        where=where_clause if where_clause else None,
        include=['documents', 'metadatas', 'distances']
    )
    
    # Post-process and rank results
    processed_results = self._post_process_results(results, query)
    
    # Return top N after processing
    return processed_results[:n_results]
```

#### Result Post-Processing

```python
def _post_process_results(self, raw_results: Dict, query: str) -> List[Dict]:
    """Post-process and re-rank search results."""
    
    processed_results = []
    query_words = set(query.lower().split())
    
    for i in range(len(raw_results['documents'][0])):
        content = raw_results['documents'][0][i]
        metadata = raw_results['metadatas'][0][i]
        distance = raw_results['distances'][0][i]
        
        # Calculate additional relevance signals
        content_words = set(content.lower().split())
        word_overlap = len(query_words.intersection(content_words))
        
        # Boost score for exact phrase matches
        phrase_boost = 1.0
        if query.lower() in content.lower():
            phrase_boost = 1.2
        
        # Calculate final score
        similarity_score = (1 - distance) * phrase_boost
        
        result = {
            'content': content,
            'metadata': metadata,
            'similarity_score': similarity_score,
            'word_overlap': word_overlap,
            'source_url': metadata['source_url'],
            'title': metadata.get('title', ''),
            'doc_type': metadata.get('doc_type', 'unknown')
        }
        
        processed_results.append(result)
    
    # Sort by relevance score
    processed_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return processed_results
```

## Memory Management

### ChromaDB Memory Optimization

```python
def optimize_chroma_memory():
    """Optimize ChromaDB memory usage."""
    
    # Configure ChromaDB settings for memory efficiency
    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",  # More memory efficient
        persist_directory=Config.CHROMA_DB_PATH,
        anonymized_telemetry=False
    )
    
    client = chromadb.PersistentClient(
        path=Config.CHROMA_DB_PATH,
        settings=chroma_settings
    )
    
    return client
```

### Application Memory Management

```python
import gc
import psutil
import os

def monitor_memory_usage():
    """Monitor and manage application memory usage."""
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    memory_usage = {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }
    
    # Log memory usage
    logger.info(f"Memory usage: RSS={memory_usage['rss']:.1f}MB, "
                f"VMS={memory_usage['vms']:.1f}MB, "
                f"Percent={memory_usage['percent']:.1f}%")
    
    # Trigger garbage collection if memory usage is high
    if memory_usage['percent'] > 80:
        logger.warning("High memory usage detected, triggering garbage collection")
        gc.collect()
        
        # Re-check after GC
        new_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after GC: {new_memory:.1f}MB")
    
    return memory_usage

# Automatic memory monitoring
def setup_memory_monitoring():
    """Setup periodic memory monitoring."""
    import threading
    
    def memory_monitor():
        while True:
            monitor_memory_usage()
            time.sleep(300)  # Check every 5 minutes
    
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()
```

## Request Processing Optimization

### Async Request Handling

```python
import asyncio
from flask import Flask
from quart import Quart  # Async Flask alternative

# For CPU-bound tasks, use thread pool
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

def process_expensive_operation_async(data):
    """Process expensive operations asynchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, expensive_operation, data)

# Example: Async RAG processing
async def process_rag_query_async(query: str) -> Dict:
    """Process RAG query asynchronously."""
    
    # Run embedding generation in thread pool
    embedding_task = asyncio.create_task(
        asyncio.to_thread(rag_engine._get_embedding, query)
    )
    
    # Run context building in parallel
    context_task = asyncio.create_task(
        asyncio.to_thread(build_conversation_context, session_id)
    )
    
    # Wait for both to complete
    embedding, context = await asyncio.gather(embedding_task, context_task)
    
    # Continue with search and generation
    search_results = await asyncio.to_thread(
        rag_engine.collection.query,
        query_embeddings=[embedding],
        n_results=5
    )
    
    return await asyncio.to_thread(
        rag_engine._generate_response, 
        query, 
        search_results
    )
```

### Request Batching

```python
class RequestBatcher:
    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.last_batch_time = time.time()
    
    async def add_request(self, request_data: Dict) -> Dict:
        """Add request to batch and process when ready."""
        
        # Add to pending batch
        future = asyncio.Future()
        self.pending_requests.append((request_data, future))
        
        # Process batch if conditions met
        should_process = (
            len(self.pending_requests) >= self.batch_size or
            time.time() - self.last_batch_time > self.timeout
        )
        
        if should_process:
            await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated requests as a batch."""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.last_batch_time = time.time()
        
        # Extract requests and futures
        requests = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        
        try:
            # Process batch efficiently
            results = await self._batch_process_requests(requests)
            
            # Set results for all futures
            for future, result in zip(futures, results):
                future.set_result(result)
                
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                future.set_exception(e)

# Usage
request_batcher = RequestBatcher()

@app.route('/api/chat', methods=['POST'])
async def api_chat():
    request_data = await request.get_json()
    result = await request_batcher.add_request(request_data)
    return jsonify(result)
```

## OpenAI API Optimization

### Request Optimization

```python
class OptimizedOpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.request_cache = {}
        self.rate_limiter = OpenAIRateLimiter()
    
    async def create_embedding_optimized(self, text: str) -> List[float]:
        """Create embedding with optimization strategies."""
        
        # Check cache first
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self.request_cache:
            return self.request_cache[cache_key]
        
        # Rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # Optimize text length
        optimized_text = self._optimize_text_for_embedding(text)
        
        # Generate embedding
        response = await self.client.embeddings.acreate(
            model="text-embedding-3-small",
            input=optimized_text
        )
        
        embedding = response.data[0].embedding
        
        # Cache result
        self.request_cache[cache_key] = embedding
        
        return embedding
    
    def _optimize_text_for_embedding(self, text: str) -> str:
        """Optimize text for embedding generation."""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long (8191 tokens for text-embedding-3-small)
        max_chars = 30000  # Conservative estimate
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated text for embedding: {len(text)} chars")
        
        return text.strip()
```

### Token Usage Optimization

```python
import tiktoken

class TokenOptimizer:
    def __init__(self, model: str = "gpt-4o"):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_context_tokens = 128000  # GPT-4o context limit
        self.max_response_tokens = Config.MAX_RESPONSE_TOKENS
    
    def optimize_context(self, context: str, query: str) -> str:
        """Optimize context to fit within token limits."""
        
        # Calculate token usage
        query_tokens = len(self.encoding.encode(query))
        context_tokens = len(self.encoding.encode(context))
        system_tokens = 500  # Estimate for system prompt
        
        # Available tokens for context
        available_tokens = (self.max_context_tokens - 
                          query_tokens - 
                          system_tokens - 
                          self.max_response_tokens)
        
        if context_tokens <= available_tokens:
            return context
        
        # Trim context intelligently
        return self._trim_context_smart(context, available_tokens)
    
    def _trim_context_smart(self, context: str, target_tokens: int) -> str:
        """Smart context trimming preserving important information."""
        
        # Split by sections
        sections = context.split('\n---\n')
        
        # Prioritize sections
        prioritized_sections = []
        for section in sections:
            section_tokens = len(self.encoding.encode(section))
            priority = self._calculate_section_priority(section)
            
            prioritized_sections.append({
                'content': section,
                'tokens': section_tokens,
                'priority': priority
            })
        
        # Sort by priority (highest first)
        prioritized_sections.sort(key=lambda x: x['priority'], reverse=True)
        
        # Include sections until token limit
        final_sections = []
        total_tokens = 0
        
        for section in prioritized_sections:
            if total_tokens + section['tokens'] <= target_tokens:
                final_sections.append(section['content'])
                total_tokens += section['tokens']
            else:
                break
        
        return '\n---\n'.join(final_sections)
    
    def _calculate_section_priority(self, section: str) -> float:
        """Calculate priority score for context section."""
        score = 0.0
        section_lower = section.lower()
        
        # Code examples get high priority
        if '```' in section:
            score += 0.4
        
        # API references get medium-high priority
        if any(word in section_lower for word in ['parameter', 'returns', 'example']):
            score += 0.3
        
        # Tutorials get medium priority
        if any(word in section_lower for word in ['step', 'tutorial', 'guide']):
            score += 0.2
        
        # Length bonus (not too short, not too long)
        if 200 <= len(section) <= 1000:
            score += 0.1
        
        return score
```

## Frontend Performance

### Static Asset Optimization

```python
# Flask configuration for static assets
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files with optimization."""
    response = send_from_directory('static', filename)
    
    # Add caching headers
    if filename.endswith(('.css', '.js', '.png', '.jpg', '.ico')):
        response.cache_control.max_age = 31536000  # 1 year
        response.cache_control.public = True
    
    # Add compression hint
    if filename.endswith(('.css', '.js')):
        response.headers['Vary'] = 'Accept-Encoding'
    
    return response
```

### Template Optimization

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Preload critical resources -->
    <link rel="preload" href="{{ url_for('static', filename='css/main.css') }}" as="style">
    <link rel="preload" href="{{ url_for('static', filename='js/main.js') }}" as="script">
    
    <!-- Critical CSS inline -->
    <style>
        /* Inline critical CSS for faster rendering */
        body { margin: 0; font-family: system-ui, sans-serif; }
        .loading { display: flex; justify-content: center; padding: 2rem; }
    </style>
    
    <!-- Non-critical CSS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
</head>
<body>
    <!-- Content -->
    
    <!-- Defer non-critical JavaScript -->
    <script defer src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

## Monitoring and Profiling

### Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'request_times': [],
            'cache_hit_rates': [],
            'db_query_times': [],
            'memory_usage': []
        }
    
    def record_request_time(self, endpoint: str, duration: float):
        """Record request processing time."""
        self.metrics['request_times'].append({
            'endpoint': endpoint,
            'duration': duration,
            'timestamp': time.time()
        })
        
        # Keep only recent metrics (last hour)
        cutoff_time = time.time() - 3600
        self.metrics['request_times'] = [
            metric for metric in self.metrics['request_times']
            if metric['timestamp'] > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict:
        """Get performance metrics summary."""
        recent_requests = self.metrics['request_times']
        
        if not recent_requests:
            return {'error': 'No recent requests'}
        
        durations = [req['duration'] for req in recent_requests]
        
        return {
            'avg_response_time': sum(durations) / len(durations),
            'min_response_time': min(durations),
            'max_response_time': max(durations),
            'p95_response_time': sorted(durations)[int(len(durations) * 0.95)],
            'total_requests': len(recent_requests),
            'requests_per_minute': len(recent_requests) / 60
        }

# Global performance monitor
perf_monitor = PerformanceMonitor()

@app.before_request
def start_request_timer():
    request.start_time = time.time()

@app.after_request
def record_request_performance(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        perf_monitor.record_request_time(request.endpoint, duration)
    
    return response
```

### Profiling Tools

```python
import cProfile
import pstats
import io

def profile_function(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # Analyze results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative').print_stats(10)
            
            logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
    
    return wrapper

# Usage
@profile_function
def expensive_rag_operation(query: str):
    return rag_engine.query(query)
```

## Load Testing

### API Load Testing

```python
import asyncio
import aiohttp
import statistics
from dataclasses import dataclass
from typing import List

@dataclass
class LoadTestResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p95_response_time: float
    requests_per_second: float
    error_rate: float

async def load_test_api(base_url: str, 
                       concurrent_users: int = 10,
                       requests_per_user: int = 5) -> LoadTestResult:
    """Perform load testing on API endpoints."""
    
    results = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for user in range(concurrent_users):
            for request_num in range(requests_per_user):
                task = asyncio.create_task(
                    make_test_request(session, base_url, f"user_{user}_req_{request_num}")
                )
                tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    successful = [r for r in responses if isinstance(r, dict) and 'error' not in r]
    failed = len(responses) - len(successful)
    
    response_times = [r['response_time'] for r in successful if 'response_time' in r]
    
    return LoadTestResult(
        total_requests=len(responses),
        successful_requests=len(successful),
        failed_requests=failed,
        avg_response_time=statistics.mean(response_times) if response_times else 0,
        p95_response_time=statistics.quantiles(response_times, n=20)[18] if response_times else 0,
        requests_per_second=len(responses) / total_time,
        error_rate=failed / len(responses) if responses else 0
    )

async def make_test_request(session: aiohttp.ClientSession, 
                          base_url: str, 
                          query_id: str) -> Dict:
    """Make individual test request."""
    start_time = time.time()
    
    try:
        async with session.post(
            f"{base_url}/api/chat",
            json={'query': f'Test query {query_id}'},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            
            response_time = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                data['response_time'] = response_time
                return data
            else:
                return {
                    'error': f'HTTP {response.status}',
                    'response_time': response_time
                }
                
    except Exception as e:
        return {
            'error': str(e),
            'response_time': time.time() - start_time
        }
```

## Performance Best Practices

### Code-Level Optimizations

#### Efficient Data Structures

```python
# Good: Use sets for membership testing
valid_doc_types = {'react', 'python', 'fastapi'}
if doc_type in valid_doc_types:  # O(1) lookup
    process_document()

# Bad: Use lists for membership testing
# valid_doc_types = ['react', 'python', 'fastapi']
# if doc_type in valid_doc_types:  # O(n) lookup

# Good: Use dict for key-value lookups
source_mapping = {
    'react': ReactProcessor(),
    'python': PythonProcessor(),
    'fastapi': FastAPIProcessor()
}
processor = source_mapping[doc_type]  # O(1) lookup

# Good: Use list comprehensions for filtering
active_sessions = [s for s in sessions if s.is_active()]

# Good: Use generators for large datasets
def process_large_dataset(items):
    for item in items:
        if should_process(item):
            yield process_item(item)
```

#### Lazy Loading

```python
class LazyRAGEngine:
    def __init__(self):
        self._openai_client = None
        self._chroma_client = None
        self._text_splitter = None
    
    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        return self._openai_client
    
    @property
    def chroma_client(self):
        """Lazy load ChromaDB client."""
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(
                path=Config.CHROMA_DB_PATH
            )
        return self._chroma_client
```

### Resource Management

```python
def with_resource_monitoring(func):
    """Decorator to monitor resource usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            end_memory = process.memory_info().rss
            end_time = time.time()
            
            # Log resource usage
            memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
            duration = end_time - start_time
            
            logger.info(f"Resource usage for {func.__name__}: "
                       f"{duration:.3f}s, {memory_delta:+.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with resource monitoring: {e}")
            raise
    
    return wrapper

@with_resource_monitoring
def process_large_document_batch(documents: List[Dict]):
    """Process documents with resource monitoring."""
    # Implementation
    pass
```

---

*Performance optimization is an ongoing process. Monitor these metrics regularly and adjust based on actual usage patterns and user feedback.*