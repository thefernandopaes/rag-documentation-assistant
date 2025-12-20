# Plano de Melhorias RAG - Portfolio para Recrutadores

## Objetivo
Otimizar o RAG Documentation Assistant para impressionar recrutadores técnicos, focando em performance, UI/UX e polish profissional.

**Meta:** Transformar de 6.5/10 para 8.5-9/10 em impressão profissional

**Tempo Estimado:** 1-2 dias (12-16 horas)

---

## Status Atual vs. Meta

| Métrica | Atual | Meta | Impacto |
|---------|-------|------|---------|
| Response Time | 13.83s ⚠️ | 3-5s ✅ | CRÍTICO |
| UI/UX Score | 5/10 🟡 | 8/10 ✅ | ALTO |
| Content Quality | 7/10 🟢 | 8/10 ✅ | MÉDIO |
| Interactive Features | 2/10 ⚠️ | 7/10 ✅ | ALTO |
| Overall Impression | 6.5/10 | 8.5-9/10 | CRÍTICO |

---

## Estratégia: 3 Prioridades em 3 Fases

### **FASE 1: Performance Optimization (4-6 horas)** 🔴 CRÍTICO
**Meta:** Reduzir response time de 13s para 3-5s

### **FASE 2: Frontend Polish (4-6 horas)** 🟡 IMPORTANTE
**Meta:** Interface profissional com syntax highlighting e copy buttons

### **FASE 3: UX Details (2-3 horas)** 🟢 DESEJÁVEL
**Meta:** Interatividade e polish final

---

# FASE 1: Performance Optimization (4-6 horas)

## Diagnóstico de Performance

### **Tarefa 1.1: Profiling e Identificação de Gargalos** ⏱️ 1-2h

**Objetivo:** Identificar onde os 13 segundos estão sendo gastos

**Implementação:**

1. **Adicionar Performance Logging Detalhado**

Arquivo: `rag_engine_async.py`

```python
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def generate_response(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Generate response with detailed performance metrics"""
    perf_metrics = {}
    total_start = time.time()

    try:
        # 1. Cache check
        cache_start = time.time()
        cache_key = f"response_{hash(query + str(conversation_history))}"
        cached_response = await self.cache.get(cache_key)
        perf_metrics['cache_check'] = time.time() - cache_start

        if cached_response:
            logger.info(f"⚡ Cache HIT - {perf_metrics['cache_check']:.3f}s")
            cached_response['cached'] = True
            cached_response['perf_metrics'] = perf_metrics
            return cached_response

        # 2. Query enhancement
        enhance_start = time.time()
        enhanced_query = query
        if self._is_self_query(query):
            enhanced_query = self._enhance_self_query(query)
        perf_metrics['query_enhancement'] = time.time() - enhance_start

        # 3. Document search (ChromaDB + embedding)
        search_start = time.time()
        relevant_docs = await self.search_documents(enhanced_query, n_results=5)
        perf_metrics['document_search'] = time.time() - search_start
        perf_metrics['embedding_generation'] = getattr(self, '_last_embedding_time', 0)
        perf_metrics['chromadb_query'] = perf_metrics['document_search'] - perf_metrics['embedding_generation']

        # 4. Context building
        context_start = time.time()
        context = self._build_context(relevant_docs)
        perf_metrics['context_building'] = time.time() - context_start

        # 5. LLM generation
        llm_start = time.time()
        response_data = await self._generate_llm_response(query, context, history_context)
        perf_metrics['llm_generation'] = time.time() - llm_start

        # 6. Post-processing
        post_start = time.time()
        sources = self._extract_sources(relevant_docs)
        perf_metrics['post_processing'] = time.time() - post_start

        # Total time
        perf_metrics['total'] = time.time() - total_start

        # Log breakdown
        logger.info(f"""
🔍 Performance Breakdown:
├─ Cache Check:         {perf_metrics['cache_check']:.3f}s
├─ Query Enhancement:   {perf_metrics['query_enhancement']:.3f}s
├─ Document Search:     {perf_metrics['document_search']:.3f}s
│  ├─ Embedding Gen:    {perf_metrics['embedding_generation']:.3f}s
│  └─ ChromaDB Query:   {perf_metrics['chromadb_query']:.3f}s
├─ Context Building:    {perf_metrics['context_building']:.3f}s
├─ LLM Generation:      {perf_metrics['llm_generation']:.3f}s
├─ Post-processing:     {perf_metrics['post_processing']:.3f}s
└─ TOTAL:               {perf_metrics['total']:.3f}s
        """)

        result = {
            'response': response_data.get('answer', response_data.get('response', '')),
            'code_examples': response_data.get('examples', response_data.get('code_examples', [])),
            'sources': sources,
            'related_questions': response_data.get('related_questions', response_data.get('related_concepts', [])),
            'response_time': perf_metrics['total'],
            'cached': False,
            'perf_metrics': perf_metrics  # Incluir métricas na resposta
        }

        # Pass through API-specific fields
        api_fields = ['endpoints', 'authentication', 'parameters', 'response_format', 'error_codes']
        for field in api_fields:
            if field in response_data:
                result[field] = response_data[field]

        await self.cache.set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise
```

2. **Adicionar Tracking de Embedding Time**

Arquivo: `rag_engine_async.py` - método `_get_embedding`

```python
async def _get_embedding(self, text: str) -> List[float]:
    """Generate embedding with performance tracking"""
    embed_start = time.time()

    try:
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding

        self._last_embedding_time = time.time() - embed_start
        logger.debug(f"Embedding generated in {self._last_embedding_time:.3f}s")

        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise
```

3. **Endpoint de Diagnóstico**

Arquivo: `routes_async.py`

```python
@router.get("/api/performance-stats")
async def get_performance_stats(
    db: AsyncSession = Depends(get_async_db),
    _admin_auth = Depends(validate_admin_key)
):
    """
    Get detailed performance statistics (admin only).

    Analyzes recent conversations to identify bottlenecks.
    """
    try:
        # Get last 10 conversations with response times
        result = await db.execute(
            select(Conversation)
            .order_by(Conversation.created_at.desc())
            .limit(10)
        )
        conversations = result.scalars().all()

        if not conversations:
            return {"message": "No conversations found"}

        # Calculate stats
        response_times = [c.response_time for c in conversations if c.response_time]

        stats = {
            "recent_queries": len(conversations),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "median_response_time": sorted(response_times)[len(response_times)//2] if response_times else 0,
            "p95_response_time": sorted(response_times)[int(len(response_times)*0.95)] if len(response_times) > 1 else 0,
            "slow_queries": [
                {
                    "query": c.user_query[:100],
                    "response_time": c.response_time,
                    "created_at": c.created_at.isoformat()
                }
                for c in conversations if c.response_time and c.response_time > 5.0
            ]
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Entregável:**
- Logs detalhados mostrando breakdown de tempo
- Endpoint `/api/performance-stats` funcionando
- Identificação clara do gargalo (provavelmente: LLM generation ou embedding)

---

### **Tarefa 1.2: Otimizar Cache Strategy** ⏱️ 1h

**Problema Atual:** Cache em arquivo (cache_manager_async.py) pode ser lento

**Solução:** Implementar cache em memória com TTL

Arquivo: `cache_manager_async.py`

```python
import asyncio
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class AsyncInMemoryCache:
    """
    In-memory cache with TTL for fast response caching.

    Performance: 0.001s vs 0.1s (file-based cache)
    """

    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        logger.info(f"In-memory cache initialized with TTL: {ttl}s, max_size: {max_size}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if datetime.now() > entry['expires_at']:
                del self._cache[key]
                logger.debug(f"Cache MISS (expired): {key[:50]}")
                return None

            logger.debug(f"Cache HIT: {key[:50]}")
            return entry['value']

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL"""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['created_at'])
                del self._cache[oldest_key]
                logger.debug(f"Cache evicted: {oldest_key[:50]}")

            self._cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=self.ttl)
            }
            logger.debug(f"Cache SET: {key[:50]}")

    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries removed")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values()
                if datetime.now() > entry['expires_at']
            )

            return {
                "total_entries": total_entries,
                "active_entries": total_entries - expired_entries,
                "expired_entries": expired_entries,
                "max_size": self.max_size,
                "ttl": self.ttl
            }
```

**Atualizar RAG Engine:**

Arquivo: `rag_engine_async.py` - `__init__`

```python
from cache_manager_async import AsyncInMemoryCache

def __init__(self):
    # Replace file cache with in-memory cache
    self.cache = AsyncInMemoryCache(
        ttl=Config.CACHE_TTL,
        max_size=1000  # Store up to 1000 responses
    )
    logger.info("Using in-memory cache for responses")
```

**Ganho Esperado:** 0.1s → 0.001s (cache check)

---

### **Tarefa 1.3: Otimizar ChromaDB Queries** ⏱️ 1h

**Problema:** ChromaDB pode estar lento para queries

**Soluções:**

1. **Reduzir n_results de 5 para 3**
   - Menos documentos = contexto menor = LLM mais rápido
   - Qualidade mantida (top 3 são os mais relevantes)

```python
# rag_engine_async.py - generate_response
relevant_docs = await self.search_documents(enhanced_query, n_results=3)  # Era 5
```

2. **Adicionar Index Optimization**

Arquivo: `rag_engine_async.py` - `__init__`

```python
def __init__(self):
    # ... existing code ...

    # Get or create collection with optimized settings
    self.collection = self.chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",  # Cosine similarity
            "hnsw:construction_ef": 200,  # Build quality
            "hnsw:search_ef": 100,  # Search quality vs speed tradeoff
            "hnsw:M": 16  # Connections per layer
        }
    )
```

3. **Paralelizar Embedding + Cache**

Se a mesma query é feita, cachear o embedding também:

```python
async def _get_embedding_cached(self, text: str) -> List[float]:
    """Get embedding with caching"""
    cache_key = f"emb_{hash(text)}"

    # Check cache first
    cached_emb = await self.cache.get(cache_key)
    if cached_emb:
        logger.debug("Embedding cache HIT")
        return cached_emb

    # Generate new
    embedding = await self._get_embedding(text)

    # Cache for 24 hours
    await self.cache.set(cache_key, embedding)

    return embedding
```

**Ganho Esperado:** ChromaDB query: 0.5s → 0.2s

---

### **Tarefa 1.4: Otimizar LLM Generation** ⏱️ 1-2h

**Problema:** LLM generation pode estar levando 8-10s

**Soluções:**

1. **Reduzir MAX_RESPONSE_TOKENS para API Queries**

Para queries de API, não precisamos de 3000 tokens sempre:

Arquivo: `rag_engine_async.py`

```python
async def _generate_llm_response(self, query: str, context: str, history: str) -> Dict[str, Any]:
    """Generate response with adaptive token limits"""
    try:
        is_api_query = self._is_api_related_query(query, context)

        # Adaptive token limit
        if is_api_query:
            max_tokens = 2000  # API responses são estruturadas, não precisam de 3000
        else:
            max_tokens = 1500  # Standard queries mais concisas

        # ... rest of code ...

        response = await self.openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,  # Adaptive
            temperature=Config.TEMPERATURE,
            response_format={"type": "json_object"}
        )
```

2. **Simplificar Prompts para Respostas Mais Concisas**

Arquivo: `rag_engine_async.py` - `_get_api_specialized_prompt`

```python
def _get_api_specialized_prompt(self) -> str:
    """Get optimized system prompt for API documentation"""
    return """You are DocRag, an expert API documentation assistant.

Response format (JSON) - BE CONCISE:
- answer: Clear technical explanation (2-3 paragraphs max, markdown)
- examples: Code examples array (max 3 languages: curl, python, javascript)
- endpoints: Relevant API endpoints (max 2)
- authentication: Auth requirements object
- parameters: Key parameters only (max 5)
- error_codes: Common errors (max 3)
- related_concepts: Array of 3 related topics

IMPORTANT: Keep responses concise and focused. Quality over quantity."""
```

3. **Streaming Response (Optional - Advanced)**

Se quisermos impressionar ainda mais, podemos implementar streaming:

```python
async def _generate_llm_response_streaming(self, query: str, context: str, history: str):
    """Generate response with streaming for faster perceived performance"""
    # This would require frontend changes to handle streaming
    # But gives impression of instant response
    pass
```

**Ganho Esperado:** LLM generation: 10s → 5-6s

---

### **Tarefa 1.5: Performance Testing & Validation** ⏱️ 1h

**Objetivo:** Validar que as otimizações funcionaram

**Script de Teste:**

Arquivo: `tests/performance_test.py`

```python
import asyncio
import time
import httpx
from statistics import mean, median

async def test_response_times(n_queries=10):
    """Test response times for multiple queries"""

    queries = [
        "How to authenticate with GitHub API?",
        "What is the Stripe payment API?",
        "How to use OpenAI chat completions?",
        "What are React hooks?",
        "How to deploy Next.js?",
        "GitHub repository creation API",
        "FastAPI async endpoints",
        "Stripe webhook handling",
        "OpenAI embeddings API",
        "React context API"
    ]

    times = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, query in enumerate(queries[:n_queries]):
            print(f"\n[{i+1}/{n_queries}] Testing: {query[:50]}...")

            start = time.time()
            response = await client.post(
                "http://127.0.0.1:8000/api/chat",
                json={"query": query},
                headers={"Cookie": f"session_id=perf_test_{i}"}
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                server_time = data.get('response_time', 0)
                cached = data.get('cached', False)

                times.append(elapsed)

                print(f"  ✓ Total: {elapsed:.2f}s | Server: {server_time:.2f}s | Cached: {cached}")

                # Show performance breakdown if available
                if 'perf_metrics' in data:
                    metrics = data['perf_metrics']
                    print(f"    - Embedding: {metrics.get('embedding_generation', 0):.3f}s")
                    print(f"    - ChromaDB: {metrics.get('chromadb_query', 0):.3f}s")
                    print(f"    - LLM: {metrics.get('llm_generation', 0):.3f}s")
            else:
                print(f"  ✗ Error: {response.status_code}")

    # Calculate statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Queries tested: {len(times)}")
    print(f"Average time: {mean(times):.2f}s")
    print(f"Median time: {median(times):.2f}s")
    print(f"Min time: {min(times):.2f}s")
    print(f"Max time: {max(times):.2f}s")
    print(f"P95 time: {sorted(times)[int(len(times)*0.95)]:.2f}s")
    print("="*60)

    # Success criteria
    avg = mean(times)
    if avg <= 3.0:
        print("✓✓✓ EXCELLENT - Target met (≤3s avg)")
    elif avg <= 5.0:
        print("✓✓ GOOD - Within acceptable range (≤5s avg)")
    elif avg <= 8.0:
        print("✓ ACCEPTABLE - Could be better (≤8s avg)")
    else:
        print("✗ NEEDS IMPROVEMENT - Too slow (>8s avg)")

    return times

if __name__ == "__main__":
    asyncio.run(test_response_times(10))
```

**Executar Teste:**

```bash
cd C:/Users/ferna/Dev/rag-documentation-assistant
python tests/performance_test.py
```

**Critérios de Sucesso FASE 1:**
- ✅ Avg response time ≤ 5s (target: 3-5s)
- ✅ P95 response time ≤ 7s
- ✅ Cache hits < 0.01s
- ✅ Performance breakdown visível nos logs

---

# FASE 2: Frontend Polish (4-6 horas)

## Objetivo
Transformar a interface de básica para profissional com:
- Markdown rendering correto
- Syntax highlighting em code blocks
- Copy buttons
- Loading states
- Visual polish

---

### **Tarefa 2.1: Setup Frontend Dependencies** ⏱️ 30min

**Instalar Libraries:**

Arquivo: `templates/chat.html` - adicionar no `<head>`:

```html
<!-- Markdown Rendering -->
<script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js"></script>

<!-- Syntax Highlighting -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/javascript.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/json.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/php.min.js"></script>

<!-- Icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
```

---

### **Tarefa 2.2: Implementar Markdown Rendering** ⏱️ 1h

**Objetivo:** Renderizar markdown corretamente (headers, code blocks, lists)

Arquivo: `static/js/chat.js`

```javascript
/**
 * Render markdown content with syntax highlighting
 */
function renderMarkdown(content) {
    // Configure marked options
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false
    });

    // Parse markdown
    const rawHtml = marked.parse(content);

    // Sanitize HTML to prevent XSS
    const cleanHtml = DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li', 'a', 'blockquote'],
        ALLOWED_ATTR: ['href', 'class', 'id']
    });

    return cleanHtml;
}

/**
 * Apply syntax highlighting to code blocks
 */
function applySyntaxHighlighting(element) {
    element.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}
```

---

### **Tarefa 2.3: Implementar Code Examples com Tabs** ⏱️ 2h

**Objetivo:** Mostrar code examples em tabs clicáveis ao invés de lista

Arquivo: `static/js/chat.js`

```javascript
/**
 * Render code examples with tabs for language switching
 */
function renderCodeExamples(examples) {
    if (!examples || examples.length === 0) return '';

    const tabsId = `tabs-${Date.now()}`;

    // Build tabs navigation
    let tabsHtml = `
        <div class="code-examples-section">
            <h4><i class="fas fa-code"></i> Code Examples</h4>
            <div class="code-tabs">
                <div class="tab-nav" role="tablist">
    `;

    examples.forEach((example, index) => {
        const lang = example.language || example.lang || 'text';
        const langLabel = lang.toUpperCase();
        const isActive = index === 0 ? 'active' : '';

        tabsHtml += `
            <button class="tab-button ${isActive}"
                    role="tab"
                    data-tab="${tabsId}-${index}"
                    onclick="switchTab('${tabsId}-${index}')">
                ${langLabel}
            </button>
        `;
    });

    tabsHtml += `
                </div>
                <div class="tab-content">
    `;

    // Build tab panels
    examples.forEach((example, index) => {
        const lang = example.language || example.lang || 'text';
        const code = example.code || '';
        const title = example.title || '';
        const description = example.description || '';
        const isActive = index === 0 ? 'active' : '';

        tabsHtml += `
            <div class="tab-panel ${isActive}"
                 id="${tabsId}-${index}"
                 role="tabpanel">
                ${title ? `<div class="code-title">${title}</div>` : ''}
                ${description ? `<div class="code-description">${description}</div>` : ''}
                <div class="code-block-wrapper">
                    <button class="copy-button" onclick="copyCode(this)" title="Copy code">
                        <i class="fas fa-copy"></i>
                    </button>
                    <pre><code class="language-${lang}">${escapeHtml(code)}</code></pre>
                </div>
            </div>
        `;
    });

    tabsHtml += `
                </div>
            </div>
        </div>
    `;

    return tabsHtml;
}

/**
 * Switch between code example tabs
 */
function switchTab(tabId) {
    // Get tab container
    const panel = document.getElementById(tabId);
    if (!panel) return;

    const container = panel.closest('.code-tabs');

    // Remove active class from all buttons and panels
    container.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    container.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

    // Add active class to clicked button and panel
    const button = container.querySelector(`[data-tab="${tabId}"]`);
    if (button) button.classList.add('active');
    panel.classList.add('active');
}

/**
 * Copy code to clipboard
 */
async function copyCode(button) {
    const codeBlock = button.nextElementSibling.querySelector('code');
    const code = codeBlock.textContent;

    try {
        await navigator.clipboard.writeText(code);

        // Visual feedback
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.classList.add('copied');

        setTimeout(() => {
            button.innerHTML = originalIcon;
            button.classList.remove('copied');
        }, 2000);
    } catch (err) {
        console.error('Failed to copy code:', err);
    }
}

/**
 * Escape HTML to prevent XSS in code blocks
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

**CSS para Tabs:**

Arquivo: `static/css/style.css`

```css
/* Code Examples Section */
.code-examples-section {
    margin: 20px 0;
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #e1e4e8;
}

.code-examples-section h4 {
    margin: 0 0 15px 0;
    color: #24292e;
    font-size: 16px;
    font-weight: 600;
}

.code-examples-section h4 i {
    color: #0366d6;
    margin-right: 8px;
}

/* Tabs Navigation */
.tab-nav {
    display: flex;
    gap: 8px;
    margin-bottom: 15px;
    border-bottom: 2px solid #e1e4e8;
    padding-bottom: 0;
}

.tab-button {
    background: none;
    border: none;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #586069;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
}

.tab-button:hover {
    color: #24292e;
    background: #f6f8fa;
}

.tab-button.active {
    color: #0366d6;
    border-bottom-color: #0366d6;
}

/* Tab Panels */
.tab-panel {
    display: none;
}

.tab-panel.active {
    display: block;
    animation: fadeIn 0.3s;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.code-title {
    font-weight: 600;
    color: #24292e;
    margin-bottom: 8px;
    font-size: 14px;
}

.code-description {
    color: #586069;
    margin-bottom: 12px;
    font-size: 13px;
}

/* Code Block Wrapper */
.code-block-wrapper {
    position: relative;
    background: #1e1e1e;
    border-radius: 6px;
    overflow: hidden;
}

.code-block-wrapper pre {
    margin: 0;
    padding: 16px;
    overflow-x: auto;
}

.code-block-wrapper code {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 13px;
    line-height: 1.5;
}

/* Copy Button */
.copy-button {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #fff;
    padding: 6px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
    z-index: 10;
}

.copy-button:hover {
    background: rgba(255, 255, 255, 0.2);
}

.copy-button.copied {
    background: #28a745;
    border-color: #28a745;
}

.copy-button i {
    font-size: 12px;
}
```

---

### **Tarefa 2.4: Implementar Loading States** ⏱️ 1h

**Objetivo:** Mostrar skeleton/spinner durante loading para melhor UX

Arquivo: `static/js/chat.js`

```javascript
/**
 * Show loading state with skeleton
 */
function showLoadingState() {
    const messagesDiv = document.getElementById('chat-messages');

    const loadingHtml = `
        <div class="message assistant-message loading-message" id="loading-state">
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="loading-skeleton">
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line short"></div>
                </div>
                <div class="loading-text">
                    <i class="fas fa-circle-notch fa-spin"></i>
                    Searching documentation...
                </div>
            </div>
        </div>
    `;

    messagesDiv.insertAdjacentHTML('beforeend', loadingHtml);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/**
 * Remove loading state
 */
function hideLoadingState() {
    const loadingElement = document.getElementById('loading-state');
    if (loadingElement) {
        loadingElement.remove();
    }
}

/**
 * Update loading state with progress
 */
function updateLoadingProgress(stage) {
    const loadingText = document.querySelector('.loading-text');
    if (!loadingText) return;

    const stages = {
        'embedding': 'Generating query embedding...',
        'searching': 'Searching documentation...',
        'generating': 'Generating response...',
        'formatting': 'Formatting response...'
    };

    const text = stages[stage] || 'Processing...';
    loadingText.innerHTML = `<i class="fas fa-circle-notch fa-spin"></i> ${text}`;
}
```

**CSS para Loading:**

Arquivo: `static/css/style.css`

```css
/* Loading Skeleton */
.loading-skeleton {
    margin-bottom: 15px;
}

.skeleton-line {
    height: 16px;
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
    border-radius: 4px;
    margin-bottom: 10px;
}

.skeleton-line.short {
    width: 60%;
}

@keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.loading-text {
    color: #666;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.loading-text i {
    color: #0366d6;
}
```

**Atualizar função sendMessage:**

```javascript
async function sendMessage() {
    const query = document.getElementById('user-input').value.trim();
    if (!query) return;

    // Add user message
    addMessage(query, 'user');
    document.getElementById('user-input').value = '';

    // Show loading
    showLoadingState();

    try {
        // Simulate progress updates (optional)
        setTimeout(() => updateLoadingProgress('embedding'), 500);
        setTimeout(() => updateLoadingProgress('searching'), 1500);
        setTimeout(() => updateLoadingProgress('generating'), 3000);

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const data = await response.json();

        // Hide loading
        hideLoadingState();

        // Display response
        displayResponse(data);

    } catch (error) {
        hideLoadingState();
        addMessage('An error occurred. Please try again.', 'assistant', true);
        console.error('Error:', error);
    }
}
```

---

### **Tarefa 2.5: Melhorar Visualização de Metadados** ⏱️ 1-2h

**Objetivo:** Mostrar endpoints, auth, parameters de forma visualmente atraente

Arquivo: `static/js/chat.js`

```javascript
/**
 * Render API metadata (endpoints, auth, parameters, error codes)
 */
function renderApiMetadata(data) {
    let html = '';

    // Endpoints
    if (data.endpoints && data.endpoints.length > 0) {
        html += `
            <div class="metadata-section">
                <h4><i class="fas fa-link"></i> Endpoints</h4>
                <div class="endpoints-grid">
        `;

        data.endpoints.forEach(endpoint => {
            const method = endpoint.method || 'GET';
            const path = endpoint.path || '/';
            const description = endpoint.description || '';

            html += `
                <div class="endpoint-card">
                    <div class="endpoint-header">
                        <span class="http-method method-${method.toLowerCase()}">${method}</span>
                        <code class="endpoint-path">${path}</code>
                    </div>
                    ${description ? `<p class="endpoint-desc">${description}</p>` : ''}
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    // Authentication
    if (data.authentication) {
        const auth = data.authentication;
        html += `
            <div class="metadata-section">
                <h4><i class="fas fa-lock"></i> Authentication</h4>
                <div class="auth-card">
                    <div class="auth-type">
                        <strong>Type:</strong> ${auth.type || 'N/A'}
                    </div>
        `;

        if (auth.methods && auth.methods.length > 0) {
            html += `
                <div class="auth-methods">
                    <strong>Methods:</strong>
                    <ul>
                        ${auth.methods.map(m => `<li>${m}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        if (auth.header) {
            html += `
                <div class="auth-header">
                    <strong>Header:</strong>
                    <code>${auth.header}</code>
                </div>
            `;
        }

        html += `
                </div>
            </div>
        `;
    }

    // Parameters
    if (data.parameters && data.parameters.length > 0) {
        html += `
            <div class="metadata-section">
                <h4><i class="fas fa-sliders-h"></i> Parameters</h4>
                <div class="parameters-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Type</th>
                                <th>Required</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        data.parameters.forEach(param => {
            const required = param.required ?
                '<span class="badge badge-required">Required</span>' :
                '<span class="badge badge-optional">Optional</span>';

            html += `
                <tr>
                    <td><code>${param.name}</code></td>
                    <td><span class="type-badge">${param.type || param.in || 'string'}</span></td>
                    <td>${required}</td>
                    <td>${param.description || '-'}</td>
                </tr>
            `;
        });

        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    // Error Codes
    if (data.error_codes && data.error_codes.length > 0) {
        html += `
            <div class="metadata-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Error Codes</h4>
                <div class="error-codes-list">
        `;

        data.error_codes.forEach(error => {
            const code = error.code || error.status_code;
            const message = error.message;
            const description = error.description || '';

            html += `
                <div class="error-code-item">
                    <div class="error-code-header">
                        <span class="error-code-badge">${code}</span>
                        <span class="error-code-message">${message}</span>
                    </div>
                    ${description ? `<p class="error-code-desc">${description}</p>` : ''}
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    return html;
}
```

**CSS para Metadados:**

Arquivo: `static/css/style.css`

```css
/* Metadata Sections */
.metadata-section {
    margin: 20px 0;
    padding: 16px;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #0366d6;
}

.metadata-section h4 {
    margin: 0 0 12px 0;
    color: #24292e;
    font-size: 15px;
    font-weight: 600;
}

.metadata-section h4 i {
    color: #0366d6;
    margin-right: 8px;
}

/* Endpoints */
.endpoints-grid {
    display: grid;
    gap: 12px;
}

.endpoint-card {
    background: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px;
}

.endpoint-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.http-method {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.method-get { background: #61affe; color: white; }
.method-post { background: #49cc90; color: white; }
.method-put { background: #fca130; color: white; }
.method-delete { background: #f93e3e; color: white; }
.method-patch { background: #50e3c2; color: white; }

.endpoint-path {
    font-family: monospace;
    font-size: 13px;
    color: #24292e;
    background: #f6f8fa;
    padding: 4px 8px;
    border-radius: 3px;
    flex: 1;
}

.endpoint-desc {
    color: #586069;
    font-size: 13px;
    margin: 8px 0 0 0;
}

/* Authentication */
.auth-card {
    background: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px;
}

.auth-type, .auth-methods, .auth-header {
    margin-bottom: 8px;
}

.auth-type strong, .auth-methods strong, .auth-header strong {
    color: #24292e;
    margin-right: 8px;
}

.auth-methods ul {
    margin: 4px 0 0 20px;
    padding: 0;
}

.auth-methods li {
    color: #586069;
    font-size: 13px;
    margin: 2px 0;
}

.auth-header code {
    background: #f6f8fa;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 12px;
}

/* Parameters Table */
.parameters-table {
    background: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    overflow: hidden;
}

.parameters-table table {
    width: 100%;
    border-collapse: collapse;
}

.parameters-table th {
    background: #f6f8fa;
    padding: 10px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    color: #24292e;
    border-bottom: 1px solid #e1e4e8;
}

.parameters-table td {
    padding: 10px 12px;
    font-size: 13px;
    color: #586069;
    border-bottom: 1px solid #f6f8fa;
}

.parameters-table tr:last-child td {
    border-bottom: none;
}

.parameters-table code {
    background: #f6f8fa;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 12px;
    color: #e36209;
}

.type-badge {
    display: inline-block;
    background: #dbedff;
    color: #0366d6;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

.badge-required {
    background: #ffeef0;
    color: #d73a49;
}

.badge-optional {
    background: #f0f6fc;
    color: #0366d6;
}

/* Error Codes */
.error-codes-list {
    background: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    overflow: hidden;
}

.error-code-item {
    padding: 12px;
    border-bottom: 1px solid #f6f8fa;
}

.error-code-item:last-child {
    border-bottom: none;
}

.error-code-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}

.error-code-badge {
    display: inline-block;
    background: #ffeef0;
    color: #d73a49;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
    font-family: monospace;
}

.error-code-message {
    color: #24292e;
    font-size: 13px;
    font-weight: 600;
}

.error-code-desc {
    color: #586069;
    font-size: 13px;
    margin: 4px 0 0 0;
}
```

---

### **Tarefa 2.6: Atualizar displayResponse() Principal** ⏱️ 30min

**Integrar todas as melhorias:**

Arquivo: `static/js/chat.js`

```javascript
/**
 * Display complete response with all enhancements
 */
function displayResponse(data) {
    const messagesDiv = document.getElementById('chat-messages');

    // Render main response (markdown)
    const responseHtml = renderMarkdown(data.response || 'No response available');

    // Render code examples (with tabs)
    const examplesHtml = renderCodeExamples(data.examples || []);

    // Render API metadata
    const metadataHtml = renderApiMetadata(data);

    // Render sources
    const sourcesHtml = renderSources(data.sources || []);

    // Render related questions
    const relatedHtml = renderRelatedQuestions(data.related_concepts || []);

    // Performance metrics (if in debug mode)
    const perfHtml = renderPerformanceMetrics(data);

    // Build complete message
    const messageHtml = `
        <div class="message assistant-message">
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="response-text">${responseHtml}</div>
                ${examplesHtml}
                ${metadataHtml}
                ${sourcesHtml}
                ${relatedHtml}
                ${perfHtml}
            </div>
        </div>
    `;

    messagesDiv.insertAdjacentHTML('beforeend', messageHtml);

    // Apply syntax highlighting to all code blocks
    applySyntaxHighlighting(messagesDiv.lastElementChild);

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/**
 * Render sources with relevance scores
 */
function renderSources(sources) {
    if (!sources || sources.length === 0) return '';

    let html = `
        <div class="sources-section">
            <h4><i class="fas fa-book"></i> Sources</h4>
            <div class="sources-list">
    `;

    sources.forEach(source => {
        const title = source.title || 'Untitled';
        const url = source.url || '#';
        const type = source.type || 'unknown';
        const relevance = source.relevance || 0;
        const relevancePercent = Math.round(relevance * 100);

        html += `
            <div class="source-item">
                <a href="${url}" target="_blank" class="source-link">
                    <i class="fas fa-external-link-alt"></i>
                    ${title}
                </a>
                <div class="source-meta">
                    <span class="source-type">${type}</span>
                    <span class="source-relevance">
                        <i class="fas fa-chart-bar"></i>
                        ${relevancePercent}% relevance
                    </span>
                </div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    return html;
}

/**
 * Render related questions as clickable chips
 */
function renderRelatedQuestions(questions) {
    if (!questions || questions.length === 0) return '';

    let html = `
        <div class="related-section">
            <h4><i class="fas fa-lightbulb"></i> Related Topics</h4>
            <div class="related-chips">
    `;

    questions.forEach(question => {
        html += `
            <button class="related-chip" onclick="askRelatedQuestion('${escapeQuotes(question)}')">
                ${question}
            </button>
        `;
    });

    html += `
            </div>
        </div>
    `;

    return html;
}

/**
 * Render performance metrics (debug mode)
 */
function renderPerformanceMetrics(data) {
    if (!data.perf_metrics) return '';

    const metrics = data.perf_metrics;
    const showDebug = localStorage.getItem('debug_mode') === 'true';

    if (!showDebug) return '';

    return `
        <div class="perf-metrics">
            <details>
                <summary>⚡ Performance Breakdown</summary>
                <ul>
                    <li>Embedding: ${(metrics.embedding_generation || 0).toFixed(3)}s</li>
                    <li>ChromaDB: ${(metrics.chromadb_query || 0).toFixed(3)}s</li>
                    <li>LLM Generation: ${(metrics.llm_generation || 0).toFixed(3)}s</li>
                    <li>Total: ${(metrics.total || 0).toFixed(3)}s</li>
                </ul>
            </details>
        </div>
    `;
}

/**
 * Handle clicking on related question
 */
function askRelatedQuestion(question) {
    document.getElementById('user-input').value = question;
    sendMessage();
}

/**
 * Escape quotes for HTML attributes
 */
function escapeQuotes(str) {
    return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
```

**CSS para Sources e Related:**

Arquivo: `static/css/style.css`

```css
/* Sources Section */
.sources-section {
    margin: 20px 0;
    padding: 16px;
    background: #f8f9fa;
    border-radius: 8px;
}

.sources-section h4 {
    margin: 0 0 12px 0;
    color: #24292e;
    font-size: 15px;
    font-weight: 600;
}

.sources-section h4 i {
    color: #0366d6;
    margin-right: 8px;
}

.sources-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.source-item {
    background: white;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px;
}

.source-link {
    color: #0366d6;
    text-decoration: none;
    font-weight: 500;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.source-link:hover {
    text-decoration: underline;
}

.source-link i {
    font-size: 11px;
}

.source-meta {
    display: flex;
    gap: 12px;
    margin-top: 6px;
    font-size: 12px;
    color: #586069;
}

.source-type {
    background: #f6f8fa;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 600;
}

.source-relevance {
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Related Questions */
.related-section {
    margin: 20px 0;
    padding: 16px;
    background: #f8f9fa;
    border-radius: 8px;
}

.related-section h4 {
    margin: 0 0 12px 0;
    color: #24292e;
    font-size: 15px;
    font-weight: 600;
}

.related-section h4 i {
    color: #ffa500;
    margin-right: 8px;
}

.related-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.related-chip {
    background: white;
    border: 1px solid #e1e4e8;
    padding: 8px 14px;
    border-radius: 16px;
    font-size: 13px;
    color: #24292e;
    cursor: pointer;
    transition: all 0.2s;
}

.related-chip:hover {
    background: #0366d6;
    color: white;
    border-color: #0366d6;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(3, 102, 214, 0.2);
}

/* Performance Metrics (Debug) */
.perf-metrics {
    margin: 20px 0;
    font-size: 12px;
    color: #666;
}

.perf-metrics details {
    background: #f6f8fa;
    padding: 10px;
    border-radius: 4px;
    cursor: pointer;
}

.perf-metrics summary {
    font-weight: 600;
    color: #24292e;
}

.perf-metrics ul {
    margin: 10px 0 0 20px;
    padding: 0;
}

.perf-metrics li {
    margin: 4px 0;
    font-family: monospace;
}
```

---

**Critérios de Sucesso FASE 2:**
- ✅ Markdown renderizando corretamente (headers, lists, code)
- ✅ Syntax highlighting em code blocks
- ✅ Code examples em tabs clicáveis
- ✅ Copy button em cada code block
- ✅ Loading state com skeleton
- ✅ Metadados (endpoints, auth, params) visualmente atraentes
- ✅ Sources clicáveis com relevance scores
- ✅ Related questions clicáveis

---

# FASE 3: UX Details & Polish (2-3 horas)

## Objetivo
Adicionar detalhes de UX que transformam um projeto bom em excelente

---

### **Tarefa 3.1: Responsive Design** ⏱️ 1h

**Objetivo:** Garantir que funcione bem em mobile

Arquivo: `static/css/style.css`

```css
/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }

    .chat-container {
        height: calc(100vh - 100px);
    }

    .tab-nav {
        flex-wrap: wrap;
    }

    .tab-button {
        font-size: 11px;
        padding: 8px 12px;
    }

    .endpoints-grid {
        grid-template-columns: 1fr;
    }

    .parameters-table {
        overflow-x: auto;
    }

    .parameters-table table {
        min-width: 500px;
    }

    .related-chips {
        flex-direction: column;
    }

    .related-chip {
        width: 100%;
        text-align: center;
    }

    .code-block-wrapper {
        font-size: 11px;
    }
}

@media (max-width: 480px) {
    .message-content {
        font-size: 14px;
    }

    .http-method {
        font-size: 10px;
    }

    .endpoint-path {
        font-size: 11px;
    }
}
```

---

### **Tarefa 3.2: Keyboard Shortcuts** ⏱️ 30min

**Objetivo:** Enter para enviar, Ctrl+L para limpar chat

Arquivo: `static/js/chat.js`

```javascript
/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    const input = document.getElementById('user-input');

    // Enter to send (Shift+Enter for newline)
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Global shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+L to clear chat
        if (e.ctrlKey && e.key === 'l') {
            e.preventDefault();
            if (confirm('Clear chat history?')) {
                clearChat();
            }
        }

        // Ctrl+/ to focus input
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            input.focus();
        }
    });
}

/**
 * Clear chat messages
 */
function clearChat() {
    const messagesDiv = document.getElementById('chat-messages');
    messagesDiv.innerHTML = '';

    // Show welcome message
    const welcomeHtml = `
        <div class="message assistant-message">
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <p>👋 Welcome! I'm your API documentation assistant.</p>
                <p>Ask me anything about:</p>
                <ul>
                    <li>GitHub API</li>
                    <li>Stripe API</li>
                    <li>OpenAI API</li>
                    <li>React documentation</li>
                    <li>Next.js documentation</li>
                    <li>FastAPI documentation</li>
                </ul>
            </div>
        </div>
    `;
    messagesDiv.innerHTML = welcomeHtml;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupKeyboardShortcuts();
});
```

---

### **Tarefa 3.3: Toast Notifications** ⏱️ 30min

**Objetivo:** Feedback visual para ações (copy, error, etc)

Arquivo: `static/js/chat.js`

```javascript
/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);

    // Remove after 3s
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
```

**CSS:**

```css
/* Toast Notifications */
.toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 12px 20px;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    opacity: 0;
    transform: translateY(20px);
    transition: all 0.3s;
    z-index: 1000;
}

.toast.show {
    opacity: 1;
    transform: translateY(0);
}

.toast-info { background: #0366d6; }
.toast-success { background: #28a745; }
.toast-error { background: #d73a49; }
.toast-warning { background: #ffa500; }
```

---

### **Tarefa 3.4: Error Handling Visual** ⏱️ 30min

**Objetivo:** Mensagens de erro bonitas

Arquivo: `static/js/chat.js`

```javascript
/**
 * Display error message
 */
function displayError(message) {
    const messagesDiv = document.getElementById('chat-messages');

    const errorHtml = `
        <div class="message error-message">
            <div class="message-avatar">
                <i class="fas fa-exclamation-circle"></i>
            </div>
            <div class="message-content">
                <div class="error-box">
                    <div class="error-header">
                        <i class="fas fa-exclamation-triangle"></i>
                        Something went wrong
                    </div>
                    <p>${message}</p>
                    <button class="retry-button" onclick="retryLastMessage()">
                        <i class="fas fa-redo"></i> Try Again
                    </button>
                </div>
            </div>
        </div>
    `;

    messagesDiv.insertAdjacentHTML('beforeend', errorHtml);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

let lastQuery = '';

async function sendMessage() {
    const query = document.getElementById('user-input').value.trim();
    if (!query) return;

    lastQuery = query;  // Save for retry

    // ... rest of sendMessage code ...

    try {
        // ... fetch logic ...
    } catch (error) {
        hideLoadingState();
        displayError('Failed to get response. Please check your connection and try again.');
        console.error('Error:', error);
    }
}

function retryLastMessage() {
    if (lastQuery) {
        document.getElementById('user-input').value = lastQuery;
        sendMessage();
    }
}
```

**CSS:**

```css
/* Error Message */
.error-message .message-avatar {
    background: #ffeef0;
    color: #d73a49;
}

.error-box {
    background: #ffeef0;
    border: 1px solid #d73a49;
    border-radius: 6px;
    padding: 16px;
}

.error-header {
    color: #d73a49;
    font-weight: 600;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.error-box p {
    color: #586069;
    margin: 8px 0;
}

.retry-button {
    background: #d73a49;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    margin-top: 8px;
}

.retry-button:hover {
    background: #cb2431;
}
```

---

**Critérios de Sucesso FASE 3:**
- ✅ Funciona bem em mobile
- ✅ Enter envia mensagem
- ✅ Ctrl+L limpa chat
- ✅ Toast notifications para feedback
- ✅ Error handling visual
- ✅ Retry button em errors

---

# Checklist Final de Implementação

## FASE 1: Performance ⏱️ 4-6h
- [ ] **1.1** Performance logging detalhado implementado
- [ ] **1.2** Cache in-memory implementado
- [ ] **1.3** ChromaDB otimizado (n_results=3, indexing)
- [ ] **1.4** LLM generation otimizado (adaptive tokens)
- [ ] **1.5** Performance test executado
- [ ] **✅ Meta**: Response time ≤ 5s (ideal: 3-5s)

## FASE 2: Frontend Polish ⏱️ 4-6h
- [ ] **2.1** Dependencies adicionadas (marked, highlight.js, DOMPurify)
- [ ] **2.2** Markdown rendering implementado
- [ ] **2.3** Code examples com tabs e copy buttons
- [ ] **2.4** Loading states com skeleton
- [ ] **2.5** API metadata visualmente atraente
- [ ] **2.6** displayResponse() atualizado
- [ ] **✅ Meta**: Interface profissional e polida

## FASE 3: UX Details ⏱️ 2-3h
- [ ] **3.1** Responsive design para mobile
- [ ] **3.2** Keyboard shortcuts (Enter, Ctrl+L)
- [ ] **3.3** Toast notifications
- [ ] **3.4** Error handling visual
- [ ] **✅ Meta**: Experiência de usuário suave

---

# Métricas de Sucesso Final

| Métrica | Antes | Meta | Como Medir |
|---------|-------|------|------------|
| Response Time (avg) | 13.83s | 3-5s | Performance test |
| Response Time (p95) | ~15s | ≤7s | Performance test |
| UI/UX Score | 5/10 | 8/10 | Manual review |
| Code Examples UX | Básico | Tabs + Copy | Manual test |
| Loading Experience | Sem feedback | Skeleton + progress | Manual test |
| Mobile Experience | Não testado | Funcional | Mobile test |
| Error Handling | Console only | Visual + retry | Trigger error |
| Overall Impression | 6.5/10 | 8.5-9/10 | Recruiter perspective |

---

# Próximos Passos Após Implementação

1. **Testing Completo:**
   - [ ] Test em Chrome, Firefox, Safari
   - [ ] Test em mobile (iOS, Android)
   - [ ] Test com diferentes queries
   - [ ] Test error scenarios

2. **Screenshots para Portfolio:**
   - [ ] Chat interface com response completa
   - [ ] Code examples com tabs
   - [ ] API metadata sections
   - [ ] Loading state
   - [ ] Mobile view

3. **Demo Video (Opcional):**
   - [ ] 30-60 segundo walkthrough
   - [ ] Mostrar performance (3-5s responses)
   - [ ] Highlight features (tabs, copy, metadata)

4. **README Update:**
   - [ ] Add screenshots
   - [ ] Performance benchmarks
   - [ ] Features list
   - [ ] Tech stack

---

# Estimativa de Tempo Total

| Fase | Tempo | Prioridade |
|------|-------|------------|
| FASE 1: Performance | 4-6h | 🔴 CRÍTICO |
| FASE 2: Frontend | 4-6h | 🟡 IMPORTANTE |
| FASE 3: UX Details | 2-3h | 🟢 DESEJÁVEL |
| **TOTAL** | **10-15h** | **1-2 dias** |

---

**Pronto para impressionar recrutadores!** 🚀
