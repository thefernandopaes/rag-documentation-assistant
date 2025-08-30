# RAG Engine Documentation

This document describes the implementation of the RAG (Retrieval-Augmented Generation) engine, including vector storage, retrieval mechanisms, and response generation.

## RAG Engine Overview

The RAG engine (`rag_engine.py`) is the core component responsible for:
- **Document embedding** and vector storage
- **Semantic search** and retrieval
- **Context building** from retrieved documents
- **Response generation** using OpenAI GPT-4o
- **Caching** for performance optimization

## Architecture

```
Query Input
    ↓
Query Embedding (OpenAI)
    ↓
Vector Search (ChromaDB)
    ↓
Context Building
    ↓
Response Generation (GPT-4o)
    ↓
Response + Sources
```

## RAG Engine Implementation

### Initialization

```python
class RAGEngine:
    def __init__(self):
        # Validate configuration
        Config.validate_config()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(allow_reset=True)
        )
        
        # Get or create collection with cosine similarity
        self.collection = self.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize cache manager
        self.cache = CacheManager()
```

### Document Processing

#### Text Chunking Strategy

The system uses LangChain's `RecursiveCharacterTextSplitter` with hierarchical separators:

```python
separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
```

**Chunking Parameters**:
- **chunk_size**: 800 characters (configurable via `CHUNK_SIZE`)
- **chunk_overlap**: 150 characters (configurable via `CHUNK_OVERLAP`)
- **separators**: Hierarchical splitting for natural boundaries

**Chunking Process**:
```python
def _chunk_document(self, content: str, metadata: dict) -> List[Dict]:
    """Split document into chunks with metadata."""
    chunks = self.text_splitter.split_text(content)
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:  # Skip tiny chunks
            continue
            
        chunk_data = {
            'content': chunk.strip(),
            'metadata': {
                **metadata,
                'chunk_index': i,
                'chunk_size': len(chunk)
            }
        }
        processed_chunks.append(chunk_data)
    
    return processed_chunks
```

#### Embedding Generation

Uses OpenAI's `text-embedding-3-small` model for high-quality embeddings:

```python
def _get_embedding(self, text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    try:
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text.replace("\n", " ")  # Normalize line breaks
        )
        return response.data[0].embedding
    
    except openai.APIError as e:
        logger.error(f"OpenAI embedding error: {e}")
        raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected embedding error: {e}")
        raise EmbeddingError(f"Embedding generation failed: {e}")
```

### Vector Storage

#### ChromaDB Integration

**Collection Configuration**:
```python
collection = chroma_client.get_or_create_collection(
    name="docrag_embeddings",
    metadata={"hnsw:space": "cosine"}  # Cosine similarity
)
```

**Document Addition**:
```python
def add_documents(self, documents: List[Dict[str, Any]]) -> None:
    """Add documents to the vector store with deduplication."""
    
    for doc in documents:
        try:
            # Generate content hash for idempotency
            content_hash = self._generate_content_hash(doc['content'])
            
            # Check if already exists
            if self._document_exists(content_hash):
                logger.info(f"Skipping duplicate document: {doc.get('title', 'Unknown')}")
                continue
            
            # Generate embedding
            embedding = self._get_embedding(doc['content'])
            
            # Create unique document ID
            doc_id = f"{doc['source_url']}#{doc['chunk_index']}"
            
            # Add to ChromaDB
            self.collection.add(
                documents=[doc['content']],
                embeddings=[embedding],
                metadatas=[{
                    'source_url': doc['source_url'],
                    'title': doc.get('title', ''),
                    'doc_type': doc.get('doc_type', 'unknown'),
                    'chunk_index': doc['chunk_index'],
                    'content_hash': content_hash
                }],
                ids=[doc_id]
            )
            
            # Store metadata in PostgreSQL
            self._store_document_metadata(doc, content_hash, doc_id)
            
            logger.info(f"Added document: {doc.get('title', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to add document {doc.get('title', 'Unknown')}: {e}")
            continue
```

#### Idempotency Management

```python
def _document_exists(self, content_hash: str) -> bool:
    """Check if document with content hash already exists."""
    from models import DocumentChunk
    existing = DocumentChunk.query.filter_by(content_hash=content_hash).first()
    return existing is not None

def _remove_existing_documents(self, source_url: str):
    """Remove existing documents for a source before adding new ones."""
    # Remove from PostgreSQL
    from models import DocumentChunk
    existing_chunks = DocumentChunk.query.filter_by(source_url=source_url).all()
    
    # Collect embedding IDs to remove from ChromaDB
    embedding_ids_to_remove = [chunk.embedding_id for chunk in existing_chunks 
                              if chunk.embedding_id]
    
    # Remove from ChromaDB
    if embedding_ids_to_remove:
        try:
            self.collection.delete(ids=embedding_ids_to_remove)
        except Exception as e:
            logger.warning(f"Failed to remove embeddings from ChromaDB: {e}")
    
    # Remove from PostgreSQL
    for chunk in existing_chunks:
        db.session.delete(chunk)
    
    db.session.commit()
    logger.info(f"Removed {len(existing_chunks)} existing chunks for {source_url}")
```

### Retrieval Mechanism

#### Semantic Search

```python
def _retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict]:
    """Retrieve most relevant document chunks for query."""
    
    # Generate query embedding
    query_embedding = self._get_embedding(query)
    
    # Search ChromaDB
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    # Process results
    retrieved_chunks = []
    for i in range(len(results['documents'][0])):
        chunk = {
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
            'source_url': results['metadatas'][0][i]['source_url'],
            'title': results['metadatas'][0][i].get('title', ''),
        }
        retrieved_chunks.append(chunk)
    
    return retrieved_chunks
```

#### Relevance Filtering

```python
def _filter_relevant_chunks(self, chunks: List[Dict], 
                           min_similarity: float = 0.7) -> List[Dict]:
    """Filter chunks based on similarity threshold."""
    relevant_chunks = []
    
    for chunk in chunks:
        if chunk['similarity_score'] >= min_similarity:
            relevant_chunks.append(chunk)
        else:
            logger.debug(f"Filtered out low-similarity chunk: "
                        f"{chunk['similarity_score']:.3f} < {min_similarity}")
    
    return relevant_chunks
```

### Context Building

#### Context Assembly

```python
def _build_context(self, retrieved_chunks: List[Dict], 
                   conversation_history: List[Dict] = None) -> str:
    """Build context from retrieved chunks and conversation history."""
    
    context_parts = []
    
    # Add conversation history (last 3 exchanges)
    if conversation_history:
        recent_history = conversation_history[-3:]  # Last 3 exchanges
        for exchange in recent_history:
            context_parts.append(f"Previous Q: {exchange['user_query']}")
            context_parts.append(f"Previous A: {exchange['ai_response'][:200]}...")
    
    # Add retrieved documentation
    context_parts.append("\n--- RELEVANT DOCUMENTATION ---")
    
    for chunk in retrieved_chunks:
        source_info = f"Source: {chunk['source_url']}"
        if chunk.get('title'):
            source_info += f" - {chunk['title']}"
        
        context_parts.append(f"\n{source_info}")
        context_parts.append(f"Content: {chunk['content']}")
        context_parts.append("---")
    
    return "\n".join(context_parts)
```

#### Token Management

```python
import tiktoken

def _estimate_tokens(self, text: str) -> int:
    """Estimate token count for text."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

def _trim_context_to_token_limit(self, context: str, 
                                max_tokens: int = 4000) -> str:
    """Trim context to fit within token limits."""
    current_tokens = self._estimate_tokens(context)
    
    if current_tokens <= max_tokens:
        return context
    
    # Trim from the beginning (oldest content first)
    lines = context.split('\n')
    trimmed_lines = []
    current_length = 0
    
    # Start from the end and work backwards
    for line in reversed(lines):
        line_tokens = self._estimate_tokens(line)
        if current_length + line_tokens > max_tokens:
            break
        trimmed_lines.insert(0, line)
        current_length += line_tokens
    
    return '\n'.join(trimmed_lines)
```

### Response Generation

#### Prompt Engineering

```python
def _generate_response(self, query: str, context: str) -> Dict[str, Any]:
    """Generate response using OpenAI GPT-4o."""
    
    system_prompt = """You are a helpful technical documentation assistant. 
    
    Guidelines:
    - Provide accurate, helpful answers based on the provided documentation
    - Include specific examples when possible
    - Reference source URLs when mentioning specific features
    - If information is not in the provided context, clearly state this
    - Keep responses concise but comprehensive
    - Format code examples with proper syntax highlighting hints
    
    Always structure your response as JSON with these fields:
    - "answer": The main response text
    - "examples": Array of code examples (if applicable)
    - "sources": Array of source URLs referenced
    - "confidence": Confidence level (high/medium/low)
    """
    
    user_prompt = f"""
    Context from documentation:
    {context}
    
    User question: {query}
    
    Please provide a helpful answer based on the context above.
    """
    
    try:
        response = self.openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=Config.MAX_RESPONSE_TOKENS,
            temperature=Config.TEMPERATURE,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        response_text = response.choices[0].message.content
        return json.loads(response_text)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return {
            "answer": response_text,
            "examples": [],
            "sources": [],
            "confidence": "low"
        }
    
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise GenerationError(f"Failed to generate response: {e}")
```

### Caching Strategy

#### Query Result Caching

```python
def query(self, query: str, conversation_history: List[Dict] = None) -> Dict:
    """Main query method with caching."""
    
    # Generate cache key
    cache_key = self._generate_cache_key(query, conversation_history)
    
    # Check cache first
    cached_result = self.cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for query: {query[:50]}...")
        return cached_result
    
    # Process query
    start_time = time.time()
    
    try:
        # Retrieve relevant chunks
        chunks = self._retrieve_relevant_chunks(query)
        
        # Build context
        context = self._build_context(chunks, conversation_history)
        
        # Generate response
        response = self._generate_response(query, context)
        
        # Add metadata
        response['response_time'] = time.time() - start_time
        response['cached'] = False
        
        # Cache result
        self.cache.set(cache_key, response, ttl=Config.CACHE_TTL)
        
        return response
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise
```

#### Cache Key Generation

```python
def _generate_cache_key(self, query: str, 
                       conversation_history: List[Dict] = None) -> str:
    """Generate cache key for query and context."""
    
    # Normalize query
    normalized_query = query.lower().strip()
    
    # Include recent history in cache key
    history_hash = ""
    if conversation_history:
        recent_history = conversation_history[-2:]  # Last 2 exchanges
        history_text = json.dumps(recent_history, sort_keys=True)
        history_hash = hashlib.md5(history_text.encode()).hexdigest()[:8]
    
    # Generate cache key
    cache_data = {
        'query': normalized_query,
        'history_hash': history_hash,
        'config_version': self._get_config_version()
    }
    
    cache_key = hashlib.sha256(
        json.dumps(cache_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    
    return f"query:{cache_key}"

def _get_config_version(self) -> str:
    """Generate version hash for RAG configuration."""
    config_params = {
        'chunk_size': Config.CHUNK_SIZE,
        'chunk_overlap': Config.CHUNK_OVERLAP,
        'max_tokens': Config.MAX_RESPONSE_TOKENS,
        'temperature': Config.TEMPERATURE,
        'model': Config.OPENAI_MODEL
    }
    
    config_str = json.dumps(config_params, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
```

## Performance Optimization

### Embedding Caching

```python
def _get_embedding_cached(self, text: str) -> List[float]:
    """Get embedding with caching for identical texts."""
    
    # Generate cache key for text
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    cache_key = f"embedding:{text_hash}"
    
    # Check cache
    cached_embedding = self.cache.get(cache_key)
    if cached_embedding:
        return cached_embedding
    
    # Generate new embedding
    embedding = self._get_embedding(text)
    
    # Cache for 24 hours (embeddings rarely change)
    self.cache.set(cache_key, embedding, ttl=86400)
    
    return embedding
```

### Batch Processing

```python
def add_documents_batch(self, documents: List[Dict[str, Any]], 
                       batch_size: int = 10) -> None:
    """Add documents in batches for better performance."""
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        try:
            # Process batch
            embeddings = []
            doc_ids = []
            metadatas = []
            contents = []
            
            for doc in batch:
                embedding = self._get_embedding(doc['content'])
                doc_id = f"{doc['source_url']}#{doc['chunk_index']}"
                
                embeddings.append(embedding)
                doc_ids.append(doc_id)
                contents.append(doc['content'])
                metadatas.append(doc['metadata'])
            
            # Batch insert to ChromaDB
            self.collection.add(
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=doc_ids
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            # Rate limiting between batches
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
            continue
```

## Error Handling

### Exception Types

```python
class RAGError(Exception):
    """Base exception for RAG engine errors."""
    pass

class EmbeddingError(RAGError):
    """Exception for embedding generation errors."""
    pass

class RetrievalError(RAGError):
    """Exception for document retrieval errors."""
    pass

class GenerationError(RAGError):
    """Exception for response generation errors."""
    pass
```

### Retry Logic

```python
import backoff

@backoff.on_exception(
    backoff.expo,
    openai.APIError,
    max_tries=Config.OPENAI_MAX_RETRIES,
    max_time=60
)
def _get_embedding_with_retry(self, text: str) -> List[float]:
    """Get embedding with exponential backoff retry."""
    return self._get_embedding(text)

@backoff.on_exception(
    backoff.expo,
    (openai.APIError, GenerationError),
    max_tries=Config.OPENAI_MAX_RETRIES,
    max_time=120
)
def _generate_response_with_retry(self, query: str, context: str) -> Dict:
    """Generate response with retry logic."""
    return self._generate_response(query, context)
```

### Graceful Degradation

```python
def query_with_fallback(self, query: str, 
                       conversation_history: List[Dict] = None) -> Dict:
    """Query with fallback strategies."""
    
    try:
        # Primary method: Full RAG pipeline
        return self.query(query, conversation_history)
        
    except EmbeddingError:
        # Fallback: Use cached embeddings if available
        logger.warning("Embedding service unavailable, using cached responses")
        return self._query_cached_only(query)
        
    except RetrievalError:
        # Fallback: Direct OpenAI without context
        logger.warning("Retrieval failed, generating response without context")
        return self._generate_direct_response(query)
        
    except GenerationError:
        # Fallback: Return template response
        logger.error("Generation failed, returning template response")
        return {
            "answer": "I'm temporarily unable to process your question. Please try again later.",
            "examples": [],
            "sources": [],
            "confidence": "low",
            "error": "service_unavailable"
        }
```

## Quality Assurance

### Response Quality Monitoring

```python
def _assess_response_quality(self, query: str, response: Dict, 
                           retrieved_chunks: List[Dict]) -> Dict:
    """Assess the quality of generated response."""
    
    quality_metrics = {
        'relevance_score': 0.0,
        'completeness_score': 0.0,
        'accuracy_indicators': []
    }
    
    # Check if response uses retrieved content
    response_text = response.get('answer', '').lower()
    chunk_texts = [chunk['content'].lower() for chunk in retrieved_chunks]
    
    # Calculate content overlap
    response_words = set(response_text.split())
    context_words = set(' '.join(chunk_texts).split())
    
    if context_words:
        overlap = len(response_words.intersection(context_words))
        quality_metrics['relevance_score'] = overlap / len(response_words)
    
    # Check for source attribution
    if response.get('sources'):
        quality_metrics['accuracy_indicators'].append('sources_provided')
    
    # Check for code examples
    if response.get('examples'):
        quality_metrics['accuracy_indicators'].append('examples_provided')
    
    # Log quality concerns
    if quality_metrics['relevance_score'] < 0.3:
        logger.warning(f"Low relevance response for query: {query[:50]}...")
    
    return quality_metrics
```

### A/B Testing Support

```python
def query_with_variants(self, query: str, variant: str = "default") -> Dict:
    """Support for A/B testing different RAG configurations."""
    
    # Store original config
    original_config = {
        'temperature': Config.TEMPERATURE,
        'max_tokens': Config.MAX_RESPONSE_TOKENS,
        'chunk_size': Config.CHUNK_SIZE
    }
    
    try:
        # Apply variant configuration
        if variant == "precise":
            Config.TEMPERATURE = 0.1
            Config.MAX_RESPONSE_TOKENS = 500
        elif variant == "comprehensive":
            Config.TEMPERATURE = 0.3
            Config.MAX_RESPONSE_TOKENS = 1000
        
        # Process query with variant settings
        result = self.query(query)
        result['variant'] = variant
        
        return result
        
    finally:
        # Restore original configuration
        for key, value in original_config.items():
            setattr(Config, key.upper(), value)
```

## Monitoring and Analytics

### Performance Metrics

```python
def get_rag_performance_stats() -> Dict:
    """Get RAG engine performance statistics."""
    
    return {
        'cache_stats': self.cache.get_stats(),
        'vector_store_stats': {
            'collection_count': len(self.chroma_client.list_collections()),
            'document_count': self.collection.count(),
        },
        'embedding_stats': {
            'model': "text-embedding-3-small",
            'dimension': 1536,
            'daily_requests': self._get_daily_embedding_requests(),
        },
        'generation_stats': {
            'model': Config.OPENAI_MODEL,
            'avg_response_time': self._get_avg_response_time(),
            'success_rate': self._get_success_rate(),
        }
    }
```

### Usage Analytics

```python
def track_query_analytics(self, query: str, response: Dict, 
                         processing_time: float) -> None:
    """Track analytics for query processing."""
    
    analytics_data = {
        'query_length': len(query),
        'response_length': len(response.get('answer', '')),
        'sources_count': len(response.get('sources', [])),
        'examples_count': len(response.get('examples', [])),
        'processing_time': processing_time,
        'confidence_level': response.get('confidence', 'unknown'),
        'cached': response.get('cached', False),
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Store analytics (could be separate analytics DB)
    logger.info(f"Query analytics: {json.dumps(analytics_data)}")
```

---

*The RAG engine is the heart of the DocRag system. Monitor its performance closely and optimize based on usage patterns and user feedback.*