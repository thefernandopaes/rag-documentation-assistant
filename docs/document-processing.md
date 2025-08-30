# Document Processing

This document describes the content crawling, extraction, and chunking strategies used in the DocRag application.

## Document Processing Overview

The Document Processor (`document_processor.py`) handles the complete pipeline for converting web-based technical documentation into searchable, embeddable chunks suitable for RAG operations.

## Processing Pipeline

```
Documentation Sources
        ↓
URL Discovery & Crawling
        ↓
Content Extraction
        ↓
Text Cleaning & Normalization
        ↓
Document Chunking
        ↓
Metadata Enrichment
        ↓
Storage (PostgreSQL + ChromaDB)
```

## DocumentProcessor Class

### Initialization

```python
class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with session and configuration."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocRag Documentation Crawler/1.0 (+https://example.com)'
        })
        self.timeout = Config.DOC_CRAWL_TIMEOUT
        
        # Configure request retry strategy
        retry_strategy = Retry(
            total=Config.REQUESTS_MAX_RETRIES,
            backoff_factor=Config.REQUESTS_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
```

## Documentation Sources

### Source Configuration

Documentation sources are defined in `config.py`:

```python
DOC_SOURCES = {
    "react": {
        "base_url": "https://react.dev/",
        "docs_url": "https://react.dev/learn",
        "type": "react"
    },
    "python": {
        "base_url": "https://docs.python.org/3/",
        "docs_url": "https://docs.python.org/3/tutorial/",
        "type": "python"
    },
    "fastapi": {
        "base_url": "https://fastapi.tiangolo.com/",
        "docs_url": "https://fastapi.tiangolo.com/tutorial/",
        "type": "fastapi"
    }
}
```

### Source Processing

```python
def process_documentation_sources(self) -> List[Dict[str, Any]]:
    """Process all configured documentation sources."""
    all_documents = []
    
    for source_name, source_config in Config.DOC_SOURCES.items():
        logger.info(f"Processing {source_name} documentation...")
        
        try:
            if Config.DOC_USE_SAMPLE:
                docs = self._process_sample_docs(source_config)
            else:
                docs = self._crawl_and_extract(source_config)
            
            all_documents.extend(docs)
            logger.info(f"Processed {len(docs)} documents from {source_name}")
            
        except Exception as e:
            logger.error(f"Error processing {source_name}: {e}")
            continue
    
    return all_documents
```

## Web Crawling

### URL Discovery

#### Sitemap Parsing

```python
def _discover_urls_from_sitemap(self, base_url: str) -> List[str]:
    """Discover URLs from XML sitemap."""
    sitemap_urls = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{base_url}/robots.txt"  # Check for sitemap references
    ]
    
    discovered_urls = set()
    
    for sitemap_url in sitemap_urls:
        try:
            response = self.session.get(sitemap_url, timeout=self.timeout)
            if response.status_code == 200:
                urls = self._parse_sitemap_xml(response.content)
                discovered_urls.update(urls)
                
        except Exception as e:
            logger.debug(f"Failed to fetch sitemap {sitemap_url}: {e}")
            continue
    
    return list(discovered_urls)

def _parse_sitemap_xml(self, xml_content: bytes) -> List[str]:
    """Parse XML sitemap and extract URLs."""
    from xml.etree import ElementTree as ET
    
    urls = []
    
    try:
        root = ET.fromstring(xml_content)
        
        # Handle sitemap namespace
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Extract URLs from <loc> elements
        for url_elem in root.findall('.//ns:loc', namespace):
            if url_elem.text:
                urls.append(url_elem.text.strip())
                
    except ET.ParseError as e:
        logger.warning(f"Failed to parse sitemap XML: {e}")
    
    return urls
```

#### Link Crawling

```python
def _discover_urls_by_crawling(self, start_url: str, max_depth: int = 2) -> List[str]:
    """Discover URLs by crawling links."""
    discovered_urls = set()
    urls_to_process = [(start_url, 0)]  # (URL, depth)
    processed_urls = set()
    
    while urls_to_process and len(discovered_urls) < Config.DOC_MAX_PAGES_PER_SOURCE:
        current_url, depth = urls_to_process.pop(0)
        
        if current_url in processed_urls or depth > max_depth:
            continue
        
        try:
            response = self._fetch_with_rate_limit(current_url)
            if response.status_code != 200:
                continue
            
            processed_urls.add(current_url)
            discovered_urls.add(current_url)
            
            # Extract links if within depth limit
            if depth < max_depth:
                links = self._extract_links(response.content, current_url)
                for link in links:
                    if self._is_documentation_url(link):
                        urls_to_process.append((link, depth + 1))
                        
        except Exception as e:
            logger.warning(f"Failed to crawl {current_url}: {e}")
            continue
    
    return list(discovered_urls)
```

### Rate-Limited Fetching

```python
def _fetch_with_rate_limit(self, url: str) -> requests.Response:
    """Fetch URL with rate limiting and retry logic."""
    
    # Apply crawl delay
    time.sleep(Config.DOC_CRAWL_DELAY)
    
    try:
        response = self.session.get(
            url, 
            timeout=self.timeout,
            allow_redirects=True
        )
        
        # Log request details
        logger.debug(f"Fetched {url}: {response.status_code} "
                    f"({len(response.content)} bytes)")
        
        return response
        
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        raise
```

## Content Extraction

### HTML to Text Conversion

The system uses **Trafilatura** for high-quality text extraction:

```python
def _extract_content(self, html_content: str, url: str) -> Dict[str, Any]:
    """Extract clean text content from HTML."""
    import trafilatura
    
    try:
        # Use Trafilatura for content extraction
        text_content = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            include_links=False,
            output_format='txt'
        )
        
        if not text_content or len(text_content.strip()) < 100:
            logger.warning(f"Insufficient content extracted from {url}")
            return None
        
        # Extract metadata
        metadata = trafilatura.extract_metadata(html_content)
        
        return {
            'content': text_content.strip(),
            'title': metadata.title if metadata else self._extract_title_fallback(html_content),
            'url': url,
            'language': metadata.language if metadata else 'en',
            'date': metadata.date if metadata else None
        }
        
    except Exception as e:
        logger.error(f"Content extraction failed for {url}: {e}")
        return None
```

### Fallback Extraction

```python
def _extract_with_beautifulsoup(self, html_content: str, url: str) -> Dict[str, Any]:
    """Fallback content extraction using BeautifulSoup."""
    from bs4 import BeautifulSoup
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract title
        title_elem = soup.find('title') or soup.find('h1')
        title = title_elem.get_text().strip() if title_elem else "Unknown"
        
        # Extract main content
        content_selectors = [
            'main', 'article', '.content', '.documentation', 
            '.docs', '.tutorial', '.guide'
        ]
        
        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break
        
        if not content_elem:
            content_elem = soup.find('body') or soup
        
        # Clean and extract text
        text_content = content_elem.get_text(separator=' ', strip=True)
        
        # Basic content quality check
        if len(text_content) < 100:
            logger.warning(f"Low quality content from {url}: {len(text_content)} chars")
            return None
        
        return {
            'content': text_content,
            'title': title,
            'url': url,
            'extraction_method': 'beautifulsoup'
        }
        
    except Exception as e:
        logger.error(f"BeautifulSoup extraction failed for {url}: {e}")
        return None
```

## Content Processing

### Text Cleaning

```python
def _clean_content(self, content: str) -> str:
    """Clean and normalize extracted content."""
    
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Max 2 consecutive newlines
    content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces
    
    # Remove navigation artifacts
    navigation_patterns = [
        r'(Skip to main content|Skip to navigation)',
        r'(Table of Contents|In this article)',
        r'(Previous|Next|Edit this page)',
        r'(Copyright|All rights reserved|Privacy Policy)',
    ]
    
    for pattern in navigation_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Remove code block artifacts
    content = re.sub(r'```\w*\n', '```\n', content)  # Clean code block headers
    content = re.sub(r'Copy to clipboard', '', content, flags=re.IGNORECASE)
    
    # Normalize unicode
    import unicodedata
    content = unicodedata.normalize('NFKC', content)
    
    return content.strip()
```

### Language Detection

```python
def _detect_content_language(self, content: str) -> str:
    """Detect content language for processing optimization."""
    # Simple heuristic-based detection
    english_indicators = ['the', 'and', 'or', 'is', 'are', 'for', 'to', 'of']
    words = content.lower().split()[:100]  # Check first 100 words
    
    english_count = sum(1 for word in words if word in english_indicators)
    
    if english_count / len(words) > 0.1:  # 10% English indicators
        return 'en'
    else:
        return 'unknown'
```

### Content Quality Assessment

```python
def _assess_content_quality(self, content: str, url: str) -> Dict[str, Any]:
    """Assess the quality of extracted content."""
    
    quality_metrics = {
        'length': len(content),
        'word_count': len(content.split()),
        'code_blocks': content.count('```'),
        'links': content.count('http'),
        'quality_score': 0.0
    }
    
    # Calculate quality score
    score = 0.0
    
    # Length bonus (optimal 500-2000 characters)
    if 500 <= quality_metrics['length'] <= 2000:
        score += 0.3
    elif quality_metrics['length'] > 100:
        score += 0.1
    
    # Technical content indicators
    tech_indicators = ['function', 'class', 'import', 'def', 'const', 'let', 'var']
    tech_count = sum(1 for indicator in tech_indicators if indicator in content.lower())
    if tech_count > 0:
        score += 0.2
    
    # Code example bonus
    if quality_metrics['code_blocks'] > 0:
        score += 0.2
    
    # Structure bonus (headers, lists)
    if any(pattern in content for pattern in ['#', '- ', '* ', '1. ']):
        score += 0.1
    
    quality_metrics['quality_score'] = min(1.0, score)
    
    # Log low quality content
    if quality_metrics['quality_score'] < 0.3:
        logger.warning(f"Low quality content from {url}: score={quality_metrics['quality_score']:.2f}")
    
    return quality_metrics
```

## Chunking Strategy

### Intelligent Chunking

The system uses LangChain's `RecursiveCharacterTextSplitter` with custom separators:

```python
def _chunk_document_intelligently(self, content: str, metadata: Dict) -> List[Dict]:
    """Intelligently chunk document preserving semantic boundaries."""
    
    # Custom separators for technical documentation
    tech_separators = [
        "\n## ",      # Markdown headers
        "\n### ",     # Subheaders
        "\n#### ",    # Sub-subheaders
        "\n\n```",    # Code blocks
        "\n\n",       # Paragraph breaks
        "\n",         # Line breaks
        ". ",         # Sentence breaks
        "! ",         # Exclamation breaks
        "? ",         # Question breaks
        "; ",         # Semicolon breaks
        ", ",         # Comma breaks
        " ",          # Word breaks
        ""            # Character breaks
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=tech_separators,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = splitter.split_text(content)
    
    # Post-process chunks
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Skip very small chunks
        if len(chunk.strip()) < 50:
            continue
        
        # Clean chunk content
        clean_chunk = self._clean_chunk_content(chunk)
        
        # Enrich with metadata
        chunk_metadata = {
            **metadata,
            'chunk_index': i,
            'chunk_size': len(clean_chunk),
            'chunk_type': self._classify_chunk_type(clean_chunk)
        }
        
        processed_chunks.append({
            'content': clean_chunk,
            'metadata': chunk_metadata
        })
    
    return processed_chunks
```

### Chunk Classification

```python
def _classify_chunk_type(self, chunk: str) -> str:
    """Classify chunk type for better retrieval."""
    chunk_lower = chunk.lower()
    
    # Code example chunk
    if '```' in chunk or 'function ' in chunk_lower or 'class ' in chunk_lower:
        return 'code'
    
    # API reference chunk
    if any(word in chunk_lower for word in ['parameter', 'returns', 'example', 'usage']):
        return 'reference'
    
    # Tutorial/guide chunk
    if any(word in chunk_lower for word in ['step', 'first', 'next', 'tutorial', 'guide']):
        return 'tutorial'
    
    # Conceptual explanation
    if any(word in chunk_lower for word in ['concept', 'overview', 'introduction', 'what is']):
        return 'concept'
    
    return 'general'
```

### Content Enhancement

```python
def _enhance_chunk_content(self, chunk: str, source_metadata: Dict) -> str:
    """Enhance chunk content with additional context."""
    
    # Add source context at the beginning
    source_context = f"From {source_metadata.get('title', 'Documentation')}"
    if source_metadata.get('doc_type'):
        source_context += f" ({source_metadata['doc_type'].title()} documentation)"
    
    # Add breadcrumb context if available
    if 'breadcrumb' in source_metadata:
        source_context += f" - {source_metadata['breadcrumb']}"
    
    enhanced_content = f"{source_context}:\n\n{chunk}"
    
    return enhanced_content
```

## Sample Data Processing

### Development Mode

For development and testing, the system can use predefined sample data:

```python
def _process_sample_docs(self, source_config: Dict) -> List[Dict[str, Any]]:
    """Process sample documentation for development."""
    from data.sample_docs import SAMPLE_REACT_DOCS, SAMPLE_PYTHON_DOCS, SAMPLE_FASTAPI_DOCS
    
    sample_mapping = {
        'react': SAMPLE_REACT_DOCS,
        'python': SAMPLE_PYTHON_DOCS,
        'fastapi': SAMPLE_FASTAPI_DOCS
    }
    
    doc_type = source_config['type']
    sample_docs = sample_mapping.get(doc_type, [])
    
    processed_docs = []
    for i, doc in enumerate(sample_docs):
        processed_doc = {
            'content': doc['content'],
            'source_url': doc['url'],
            'title': doc['title'],
            'doc_type': doc_type,
            'chunk_index': i,
            'is_sample': True
        }
        processed_docs.append(processed_doc)
    
    return processed_docs
```

### Sample Data Format

```python
# data/sample_docs.py
SAMPLE_REACT_DOCS = [
    {
        'title': 'Your First Component',
        'url': 'https://react.dev/learn/your-first-component',
        'content': '''Components are one of the core concepts of React. They are the foundation upon which you build user interfaces (UI), which makes them the perfect place to start your React journey!

Components: UI building blocks
React applications are made out of components. A component is a piece of the UI (user interface) that has its own logic and appearance. A component can be as small as a button, or as large as an entire page.

React components are JavaScript functions that return markup:

```javascript
function MyButton() {
  return (
    <button>I'm a button</button>
  );
}
```'''
    },
    {
        'title': 'Describing the UI',
        'url': 'https://react.dev/learn/describing-the-ui',
        'content': '''React is a JavaScript library for building user interfaces. At its core, React lets you describe your UI using components.

What is JSX?
JSX is a syntax extension for JavaScript that lets you write HTML-like markup inside a JavaScript file. Although there are other ways to write components, most React developers prefer the conciseness of JSX, and most codebases use it.

JSX is stricter than HTML. You have to close tags like <br />. Your component also can't return multiple JSX tags. You have to wrap them in a shared parent, like a <div>...</div> or an empty <>...</> wrapper.'''
    }
]
```

## Error Handling

### Robust Error Recovery

```python
def _process_url_with_fallbacks(self, url: str) -> Dict[str, Any]:
    """Process URL with multiple fallback strategies."""
    
    extraction_methods = [
        ('trafilatura', self._extract_with_trafilatura),
        ('beautifulsoup', self._extract_with_beautifulsoup),
        ('requests_html', self._extract_with_requests_html)
    ]
    
    for method_name, extraction_func in extraction_methods:
        try:
            logger.debug(f"Trying {method_name} extraction for {url}")
            
            response = self._fetch_with_rate_limit(url)
            if response.status_code != 200:
                continue
            
            result = extraction_func(response.content, url)
            if result and len(result.get('content', '')) > 100:
                result['extraction_method'] = method_name
                logger.info(f"Successfully extracted content from {url} using {method_name}")
                return result
                
        except Exception as e:
            logger.warning(f"{method_name} extraction failed for {url}: {e}")
            continue
    
    logger.error(f"All extraction methods failed for {url}")
    return None
```

### Content Validation

```python
def _validate_extracted_content(self, content_data: Dict) -> bool:
    """Validate extracted content quality."""
    
    if not content_data or not content_data.get('content'):
        return False
    
    content = content_data['content']
    
    # Minimum length requirement
    if len(content) < 100:
        logger.debug(f"Content too short: {len(content)} characters")
        return False
    
    # Check for extraction artifacts
    artifacts = [
        'page not found',
        '404 error',
        'access denied',
        'javascript required',
        'cookies required'
    ]
    
    content_lower = content.lower()
    if any(artifact in content_lower for artifact in artifacts):
        logger.debug(f"Content contains extraction artifacts")
        return False
    
    # Check content-to-noise ratio
    meaningful_chars = len(re.sub(r'[^\w\s]', '', content))
    if meaningful_chars / len(content) < 0.6:  # 60% meaningful characters
        logger.debug(f"Low content quality: {meaningful_chars/len(content):.2f} ratio")
        return False
    
    return True
```

## Batch Processing

### Concurrent Processing

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

async def process_urls_concurrently(self, urls: List[str], 
                                  max_concurrent: int = 5) -> List[Dict]:
    """Process multiple URLs concurrently."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_url(session, url):
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return self._extract_content(html_content, url)
            except Exception as e:
                logger.warning(f"Async processing failed for {url}: {e}")
                return None
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_single_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    successful_results = [r for r in results if r and not isinstance(r, Exception)]
    
    logger.info(f"Processed {len(successful_results)}/{len(urls)} URLs successfully")
    return successful_results
```

### Progress Tracking

```python
def process_with_progress(self, urls: List[str]) -> List[Dict]:
    """Process URLs with progress tracking."""
    
    processed_docs = []
    total_urls = len(urls)
    
    for i, url in enumerate(urls, 1):
        try:
            logger.info(f"Processing {i}/{total_urls}: {url}")
            
            doc_data = self._process_url_with_fallbacks(url)
            if doc_data:
                processed_docs.append(doc_data)
                logger.info(f"✓ Processed {url} ({len(doc_data['content'])} chars)")
            else:
                logger.warning(f"✗ Failed to process {url}")
            
            # Progress report every 10 documents
            if i % 10 == 0:
                success_rate = len(processed_docs) / i * 100
                logger.info(f"Progress: {i}/{total_urls} ({success_rate:.1f}% success rate)")
            
        except KeyboardInterrupt:
            logger.info(f"Processing interrupted. Processed {len(processed_docs)} documents.")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            continue
    
    return processed_docs
```

## Content Deduplication

### Hash-Based Deduplication

```python
import hashlib
import json

def _generate_content_hash(self, content: str, metadata: Dict = None) -> str:
    """Generate unique hash for content deduplication."""
    
    # Normalize content for hashing
    normalized_content = re.sub(r'\s+', ' ', content.strip().lower())
    
    # Include relevant metadata in hash
    hash_data = {
        'content': normalized_content,
        'url': metadata.get('source_url', '') if metadata else '',
        'doc_type': metadata.get('doc_type', '') if metadata else ''
    }
    
    # Generate SHA-256 hash
    hash_input = json.dumps(hash_data, sort_keys=True)
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

def _check_content_exists(self, content_hash: str) -> bool:
    """Check if content with hash already exists."""
    from models import DocumentChunk
    existing = DocumentChunk.query.filter_by(content_hash=content_hash).first()
    return existing is not None
```

### Similarity-Based Deduplication

```python
def _check_content_similarity(self, new_content: str, 
                            existing_contents: List[str],
                            threshold: float = 0.9) -> bool:
    """Check if content is too similar to existing content."""
    from difflib import SequenceMatcher
    
    for existing_content in existing_contents:
        similarity = SequenceMatcher(None, new_content.lower(), 
                                   existing_content.lower()).ratio()
        
        if similarity >= threshold:
            logger.debug(f"High similarity detected: {similarity:.3f}")
            return True
    
    return False
```

## Performance Optimization

### Caching Strategies

```python
def _cache_extracted_content(self, url: str, content_data: Dict):
    """Cache extracted content to avoid re-processing."""
    cache_key = f"extracted:{hashlib.md5(url.encode()).hexdigest()}"
    self.cache.set(cache_key, content_data, ttl=86400)  # Cache for 24 hours

def _get_cached_content(self, url: str) -> Dict:
    """Retrieve cached extracted content."""
    cache_key = f"extracted:{hashlib.md5(url.encode()).hexdigest()}"
    return self.cache.get(cache_key)
```

### Memory Management

```python
def _process_large_document_set(self, urls: List[str]) -> List[Dict]:
    """Process large document sets with memory management."""
    
    processed_docs = []
    batch_size = 50  # Process in batches to manage memory
    
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        
        logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_urls)} URLs")
        
        # Process batch
        batch_results = []
        for url in batch_urls:
            result = self._process_url_with_fallbacks(url)
            if result:
                batch_results.append(result)
        
        processed_docs.extend(batch_results)
        
        # Memory cleanup between batches
        import gc
        gc.collect()
        
        logger.info(f"Batch completed: {len(batch_results)} documents processed")
    
    return processed_docs
```

## Monitoring and Analytics

### Processing Metrics

```python
def get_processing_stats(self) -> Dict:
    """Get document processing statistics."""
    from models import DocumentChunk
    
    # Database statistics
    total_chunks = DocumentChunk.query.count()
    doc_types = db.session.query(
        DocumentChunk.doc_type, 
        db.func.count(DocumentChunk.id)
    ).group_by(DocumentChunk.doc_type).all()
    
    # Source statistics
    source_stats = db.session.query(
        DocumentChunk.source_url,
        db.func.count(DocumentChunk.id),
        db.func.avg(db.func.length(DocumentChunk.content))
    ).group_by(DocumentChunk.source_url).all()
    
    return {
        'total_chunks': total_chunks,
        'by_doc_type': dict(doc_types),
        'by_source': [
            {
                'url': url,
                'chunk_count': count,
                'avg_chunk_size': round(avg_size, 2)
            }
            for url, count, avg_size in source_stats
        ],
        'last_updated': DocumentChunk.query.order_by(
            DocumentChunk.updated_at.desc()
        ).first().updated_at if total_chunks > 0 else None
    }
```

### Quality Metrics

```python
def analyze_content_quality(self) -> Dict:
    """Analyze overall content quality metrics."""
    from models import DocumentChunk
    
    chunks = DocumentChunk.query.all()
    
    if not chunks:
        return {'error': 'No chunks found'}
    
    # Calculate quality metrics
    total_chunks = len(chunks)
    total_content_length = sum(len(chunk.content) for chunk in chunks)
    
    # Chunk size distribution
    chunk_sizes = [len(chunk.content) for chunk in chunks]
    
    # Content type distribution
    content_types = {}
    for chunk in chunks:
        chunk_type = self._classify_chunk_type(chunk.content)
        content_types[chunk_type] = content_types.get(chunk_type, 0) + 1
    
    return {
        'total_chunks': total_chunks,
        'avg_chunk_size': total_content_length / total_chunks,
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'content_type_distribution': content_types,
        'quality_score': self._calculate_overall_quality_score(chunks)
    }
```

---

*Document processing is a critical component that determines the quality of the RAG system. Monitor processing quality and adjust parameters based on retrieval performance.*