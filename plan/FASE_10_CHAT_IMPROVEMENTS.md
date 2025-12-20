# FASE 10 - Chat Interface & RAG Improvements

**Status**: 🟡 Planejamento
**Prioridade**: 🔴 ALTA
**Duração Estimada**: 2-3 semanas
**Data de Início**: 2025-12-19

---

## 📋 Sumário Executivo

### Objetivo
Melhorar significativamente a experiência do usuário no chat e a qualidade das respostas do sistema RAG, implementando renderização avançada, self-awareness contextual, e otimizações de UX.

### Problemas Identificados
1. ❌ JSON sendo exibido como texto bruto (sem formatação)
2. ❌ IA não reconhece perguntas sobre o próprio sistema ("this API")
3. ❌ Sem syntax highlighting nos exemplos de código
4. ❌ Fontes não são clicáveis
5. ❌ Campo `related_concepts` truncado
6. ❌ Exemplos genéricos em vez de específicos

### Métricas de Sucesso
- ✅ 100% dos códigos com syntax highlighting
- ✅ Tempo de resposta < 2s para queries simples
- ✅ 90%+ de acurácia em perguntas sobre o próprio sistema
- ✅ Taxa de satisfação de usuários > 85%
- ✅ Zero bugs de renderização

---

## 🎯 Fases de Implementação

### **FASE 10.1 - Frontend: Chat UI Enhancements** (Semana 1)
**Responsável**: @frontend-developer
**Prioridade**: 🔴 CRÍTICA
**Duração**: 5 dias

#### Objetivos
- Transformar resposta JSON em UI rica e interativa
- Adicionar syntax highlighting para código
- Tornar fontes clicáveis
- Melhorar feedback visual

#### Tasks

##### TASK 10.1.1 - Refatorar Renderização de Respostas
**Arquivo**: `static/js/chat.js`
**Responsável**: @frontend-developer
**Prioridade**: 🔴 Alta
**Estimativa**: 2 dias

**Descrição**:
Substituir a renderização atual (JSON bruto) por componentes estruturados.

**Implementação**:
```javascript
// ANTES (atual)
function displayResponse(data) {
    const response = data.response || data.answer;
    messagesDiv.innerHTML += `<div>${response}</div>`;
}

// DEPOIS (melhorado)
function displayResponse(data) {
    const responseHTML = `
        <div class="ai-message">
            <!-- Resposta principal -->
            <div class="answer-section">
                ${formatMarkdown(data.answer || data.response)}
            </div>

            <!-- Exemplos de código -->
            ${renderCodeExamples(data.examples)}

            <!-- Endpoints -->
            ${renderEndpoints(data.endpoints)}

            <!-- Autenticação -->
            ${renderAuthentication(data.authentication)}

            <!-- Fontes clicáveis -->
            ${renderSources(data.sources)}

            <!-- Conceitos relacionados -->
            ${renderRelatedConcepts(data.related_concepts)}
        </div>
    `;
    messagesDiv.innerHTML += responseHTML;

    // Aplicar syntax highlighting
    Prism.highlightAllUnder(messagesDiv);
}
```

**Entregáveis**:
1. Função `renderCodeExamples()` - Renderiza exemplos com Prism.js
2. Função `renderSources()` - Links clicáveis com ícones
3. Função `renderEndpoints()` - Tabela formatada de endpoints
4. Função `renderAuthentication()` - Card visual de autenticação
5. Função `renderRelatedConcepts()` - Tags/chips de conceitos

**Critérios de Aceitação**:
- [ ] Todo código tem syntax highlighting
- [ ] Fontes são clicáveis e abrem em nova aba
- [ ] Responsivo em mobile
- [ ] Suporta markdown na resposta principal
- [ ] Sem warnings no console

---

##### TASK 10.1.2 - Implementar Syntax Highlighting Avançado
**Arquivo**: `static/js/chat.js`, `templates/chat.html`
**Responsável**: @frontend-developer
**Prioridade**: 🔴 Alta
**Estimativa**: 1 dia

**Descrição**:
Configurar Prism.js para suportar múltiplas linguagens e plugins.

**Implementação**:
```html
<!-- Em chat.html - Adicionar plugins Prism.js -->
<link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/toolbar/prism-toolbar.min.css" rel="stylesheet">

<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/copy-to-clipboard/prism-copy-to-clipboard.min.js"></script>
```

```javascript
// Função para renderizar código com linguagem específica
function renderCodeBlock(code, language) {
    const languageClass = `language-${language.toLowerCase()}`;
    return `
        <div class="code-example">
            <div class="code-header">
                <span class="language-badge">${language}</span>
                <button class="copy-button" onclick="copyCode(this)">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>
            <pre class="line-numbers"><code class="${languageClass}">${escapeHtml(code)}</code></pre>
        </div>
    `;
}
```

**Entregáveis**:
1. Suporte a Python, JavaScript, Bash, JSON, cURL, YAML
2. Botão "Copy to Clipboard" em cada bloco de código
3. Numeração de linhas
4. Tema consistente com UI dark

**Critérios de Aceitação**:
- [ ] Todas as linguagens renderizam corretamente
- [ ] Botão de copiar funciona
- [ ] Tema dark aplicado
- [ ] Performance: < 100ms para renderizar bloco de código

---

##### TASK 10.1.3 - UI Components para Seções da Resposta
**Arquivo**: `static/js/chat.js`, `static/css/style.css`
**Responsável**: @frontend-developer
**Prioridade**: 🟡 Média
**Estimativa**: 1.5 dias

**Descrição**:
Criar componentes visuais para cada tipo de informação na resposta.

**Componentes**:

1. **Endpoints Card**:
```javascript
function renderEndpoints(endpoints) {
    if (!endpoints || endpoints.length === 0) return '';

    return `
        <div class="endpoints-section">
            <h4><i class="fas fa-plug"></i> API Endpoints</h4>
            <table class="endpoints-table">
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>Path</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    ${endpoints.map(ep => `
                        <tr>
                            <td><span class="method-badge ${ep.method.toLowerCase()}">${ep.method}</span></td>
                            <td><code>${ep.path}</code></td>
                            <td>${ep.description}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}
```

2. **Sources Card**:
```javascript
function renderSources(sources) {
    if (!sources || sources.length === 0) return '';

    return `
        <div class="sources-section">
            <h4><i class="fas fa-book"></i> Sources</h4>
            <div class="sources-list">
                ${sources.map(source => `
                    <a href="${source.url}" target="_blank" class="source-card">
                        <div class="source-icon">
                            <i class="fas fa-external-link-alt"></i>
                        </div>
                        <div class="source-content">
                            <div class="source-title">${source.title}</div>
                            <div class="source-meta">
                                <span class="source-type">${source.type}</span>
                                <span class="source-relevance">${Math.round(source.relevance * 100)}% relevant</span>
                            </div>
                        </div>
                    </a>
                `).join('')}
            </div>
        </div>
    `;
}
```

3. **Authentication Card**:
```javascript
function renderAuthentication(auth) {
    if (!auth) return '';

    return `
        <div class="auth-section">
            <h4><i class="fas fa-lock"></i> Authentication</h4>
            <div class="auth-card">
                <div class="auth-type">
                    <strong>Type:</strong> ${auth.type}
                </div>
                ${auth.flow ? `<div class="auth-flow"><strong>Flow:</strong> ${auth.flow}</div>` : ''}
                ${auth.token_url ? `<div class="auth-url"><strong>Token URL:</strong> <code>${auth.token_url}</code></div>` : ''}
            </div>
        </div>
    `;
}
```

4. **Related Concepts Tags**:
```javascript
function renderRelatedConcepts(concepts) {
    if (!concepts || concepts.length === 0) return '';

    return `
        <div class="related-concepts-section">
            <h4><i class="fas fa-tags"></i> Related Concepts</h4>
            <div class="concepts-tags">
                ${concepts.map(concept => `
                    <span class="concept-tag">${concept}</span>
                `).join('')}
            </div>
        </div>
    `;
}
```

**Entregáveis**:
1. 4 componentes visuais funcionais
2. CSS responsivo para cada componente
3. Ícones Font Awesome
4. Animações suaves de entrada

**Critérios de Aceitação**:
- [ ] Todos os componentes renderizam corretamente
- [ ] Design consistente com tema dark
- [ ] Responsivo em mobile/tablet/desktop
- [ ] Acessível (ARIA labels)

---

##### TASK 10.1.4 - Melhorar Feedback Visual e Loading States
**Arquivo**: `static/js/chat.js`, `static/css/style.css`
**Responsável**: @frontend-developer
**Prioridade**: 🟡 Média
**Estimativa**: 0.5 dia

**Descrição**:
Adicionar indicadores de loading, animações e feedback de erro.

**Implementação**:
```javascript
// Loading indicator com animação
function showTypingIndicator() {
    const indicator = `
        <div class="typing-indicator" id="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span class="typing-text">AI is thinking...</span>
        </div>
    `;
    messagesDiv.insertAdjacentHTML('beforeend', indicator);
    scrollToBottom();
}

// Error state melhorado
function showError(message) {
    const errorHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-circle"></i>
            <div class="error-content">
                <strong>Error</strong>
                <p>${message}</p>
            </div>
            <button onclick="retryLastQuery()" class="retry-button">
                <i class="fas fa-redo"></i> Retry
            </button>
        </div>
    `;
    messagesDiv.insertAdjacentHTML('beforeend', errorHTML);
}

// Success animation
function animateResponse(element) {
    element.classList.add('fade-in-up');
}
```

**CSS Animations**:
```css
@keyframes fade-in-up {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.typing-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 15px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: var(--primary-color);
    border-radius: 50%;
    animation: typing 1.4s infinite;
}

@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
}
```

**Entregáveis**:
1. Loading indicator animado
2. Error state com botão de retry
3. Success animations
4. Progress bar para requests longos

**Critérios de Aceitação**:
- [ ] Loading aparece durante requests
- [ ] Animações suaves (60 FPS)
- [ ] Retry funciona corretamente
- [ ] Sem flash de conteúdo

---

### **FASE 10.2 - Backend: Self-Awareness & Context** (Semana 1-2)
**Responsável**: @rag-specialist + @backend-architect
**Prioridade**: 🔴 CRÍTICA
**Duração**: 6 dias

#### Objetivos
- Implementar detecção de perguntas sobre o próprio sistema
- Adicionar documentação interna ao ChromaDB
- Melhorar prompts do sistema
- Corrigir bug de `related_concepts` truncado

---

##### TASK 10.2.1 - Criar Documentação Interna do Sistema
**Arquivo**: `data/internal_docs.py` (novo)
**Responsável**: @rag-specialist
**Prioridade**: 🔴 Alta
**Estimativa**: 2 dias

**Descrição**:
Criar documentação completa sobre o próprio sistema RAG para ser indexada no ChromaDB.

**Estrutura**:
```python
# data/internal_docs.py

INTERNAL_DOCUMENTATION = {
    "api_authentication": {
        "title": "RAG Documentation Assistant - Authentication",
        "content": """
# Authentication for RAG Documentation Assistant API

This API uses **session-based authentication** with cookies. No OAuth2 or API keys are required for basic usage.

## How It Works

1. **Automatic Session Creation**: When you make your first request to `/api/chat`, a session cookie is automatically created and stored in your browser.

2. **Session ID**: Each session gets a unique ID (UUID v4) that tracks your conversation history.

3. **No Manual Authentication**: You don't need to send any authentication headers for standard chat queries.

## API Endpoints

### POST /api/chat
- **Authentication**: Session cookie (automatic)
- **Rate Limit**: 10 requests/minute
- **Payload**:
  ```json
  {
    "query": "your question here",
    "session_id": "optional-custom-session-id"
  }
  ```

### GET /api/history
- **Authentication**: Session cookie
- **Returns**: Your conversation history

### POST /api/feedback
- **Authentication**: Session cookie
- **Rate Limit**: 5 requests/minute

## Admin Endpoints

For admin-only endpoints like `/api/initialize`, you need to provide an admin API key:

```bash
curl -X POST http://localhost:8000/api/initialize \\
  -H "X-Admin-Key: your-admin-key-here" \\
  -H "Content-Type: application/json"
```

## Security Features

- HTTPS enforced in production
- CORS configured for allowed domains
- Request size limits (16KB)
- Rate limiting per session
- XSS protection headers
        """,
        "type": "internal",
        "url": "internal://authentication"
    },

    "api_endpoints": {
        "title": "RAG Documentation Assistant - Available Endpoints",
        "content": """
# API Endpoints Reference

## Chat Endpoints

### POST /api/chat
Generate AI responses using RAG.

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
  "response": "AI generated answer...",
  "sources": [...],
  "examples": [...],
  "response_time": 2.5,
  "cached": false
}
```

## History Endpoints

### GET /api/history?limit=10
Get conversation history for current session.

### GET /api/stats
Get system statistics (documents, conversations, cache).

## Frontend Routes

- `GET /` - Homepage
- `GET /chat` - Chat interface
- `GET /docs` - API documentation (Swagger UI)
        """,
        "type": "internal",
        "url": "internal://endpoints"
    },

    "system_architecture": {
        "title": "RAG Documentation Assistant - System Architecture",
        "content": """
# System Architecture

## Technology Stack

- **Framework**: FastAPI (async)
- **Database**: SQLite (async with aiosqlite)
- **Vector DB**: ChromaDB
- **LLM**: OpenAI GPT-4
- **Embeddings**: text-embedding-3-small (1536 dimensions)
- **Frontend**: Vanilla JS + Bootstrap + Prism.js

## RAG Pipeline

1. **Query Processing**:
   - User submits query
   - Query validation (XSS, SQL injection protection)
   - Rate limiting check

2. **Embedding Generation**:
   - Query → OpenAI embeddings API
   - 1536-dimensional vector

3. **Document Retrieval**:
   - ChromaDB semantic search (cosine similarity)
   - Top 5 most relevant chunks
   - Metadata filtering

4. **Response Generation**:
   - Context: retrieved chunks + conversation history
   - OpenAI GPT-4 with custom system prompt
   - Enhanced with code examples, endpoints, etc.

5. **Post-Processing**:
   - Save conversation to database
   - Update cache
   - Return structured response
        """,
        "type": "internal",
        "url": "internal://architecture"
    },

    "code_examples": {
        "title": "RAG Documentation Assistant - Usage Examples",
        "content": """
# How to Use This API

## Python Example

```python
import requests

# Make a query
response = requests.post('http://localhost:8000/api/chat', json={
    'query': 'How to handle errors in FastAPI?'
})

data = response.json()
print(f"Answer: {data['response']}")
print(f"Sources: {len(data['sources'])} found")
```

## JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        query: 'What is async/await in Python?'
    })
});

const data = await response.json();
console.log(data.response);
```

## cURL Example

```bash
curl -X POST http://localhost:8000/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Explain FastAPI dependencies"}'
```
        """,
        "type": "internal",
        "url": "internal://examples"
    }
}

def get_all_internal_docs():
    """Return all internal documentation as list of dicts"""
    return [
        {
            "title": doc["title"],
            "content": doc["content"],
            "metadata": {
                "source": "internal",
                "type": doc["type"],
                "url": doc["url"]
            }
        }
        for doc in INTERNAL_DOCUMENTATION.values()
    ]
```

**Entregáveis**:
1. Arquivo `data/internal_docs.py` completo
2. Documentação de autenticação
3. Documentação de endpoints
4. Documentação de arquitetura
5. Exemplos de uso em 3 linguagens

**Critérios de Aceitação**:
- [ ] Documentação completa e precisa
- [ ] Formato compatível com ChromaDB
- [ ] Exemplos testados e funcionais
- [ ] Cobertura de 100% dos endpoints públicos

---

##### TASK 10.2.2 - Indexar Documentação Interna no ChromaDB
**Arquivo**: `document_processor.py`, `rag_engine_async.py`
**Responsável**: @rag-specialist
**Prioridade**: 🔴 Alta
**Estimativa**: 1 dia

**Descrição**:
Modificar o sistema de inicialização para incluir documentação interna.

**Implementação**:
```python
# Em document_processor.py

async def initialize_internal_docs(rag_engine):
    """Initialize internal documentation in ChromaDB"""
    from data.internal_docs import get_all_internal_docs

    logger.info("Initializing internal documentation...")

    internal_docs = get_all_internal_docs()

    for doc in internal_docs:
        # Criar chunks
        chunks = create_chunks(doc["content"], chunk_size=800, overlap=150)

        # Adicionar ao ChromaDB
        for i, chunk in enumerate(chunks):
            await rag_engine.add_document(
                text=chunk,
                metadata={
                    **doc["metadata"],
                    "chunk_id": i,
                    "title": doc["title"],
                    "is_internal": True
                }
            )

    logger.info(f"Indexed {len(internal_docs)} internal documents")
```

**Mudanças em `routes_async.py`**:
```python
@router.post("/api/initialize")
async def initialize_documents(
    request: InitializeRequest,
    rag_engine = Depends(get_rag_engine),
    _admin: None = Depends(validate_admin_key)
):
    # ... código existente ...

    # ADICIONAR: Indexar documentação interna
    await initialize_internal_docs(rag_engine)

    return InitializeResponse(...)
```

**Entregáveis**:
1. Função `initialize_internal_docs()`
2. Integração com endpoint `/api/initialize`
3. Metadata flag `is_internal: true`
4. Logs de indexação

**Critérios de Aceitação**:
- [ ] Docs internos indexados sem erros
- [ ] Metadata corretamente atribuída
- [ ] Busca semântica funciona com docs internos
- [ ] Não interfere com docs externos

---

##### TASK 10.2.3 - Implementar Detecção de Self-Queries
**Arquivo**: `rag_engine_async.py`
**Responsável**: @backend-architect
**Prioridade**: 🔴 Alta
**Estimativa**: 1.5 dias

**Descrição**:
Detectar quando o usuário pergunta sobre "this API" e priorizar documentação interna.

**Implementação**:
```python
# Em rag_engine_async.py

SELF_QUERY_KEYWORDS = [
    "this api",
    "this system",
    "this application",
    "this service",
    "your api",
    "your system",
    "how do i use",
    "how to authenticate",
    "how to use this",
    "api key",
    "session",
    "endpoints available",
    "what endpoints"
]

def is_self_query(query: str) -> bool:
    """Detect if query is about the system itself"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in SELF_QUERY_KEYWORDS)

async def search_documents(
    self,
    query: str,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Search with self-awareness.
    """
    # Detectar se é self-query
    is_self = is_self_query(query)

    if is_self:
        logger.info(f"Detected self-query: {query}")

        # Buscar APENAS em docs internos
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[await self._get_embedding(query)],
            n_results=n_results,
            where={"is_internal": True}  # Filtrar apenas internos
        )
    else:
        # Busca normal
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[await self._get_embedding(query)],
            n_results=n_results
        )

    return results
```

**System Prompt Enhancement**:
```python
SYSTEM_PROMPT_SELF_AWARE = """
You are the AI assistant for the RAG Documentation Assistant API.

IMPORTANT: When users ask about "this API", "this system", "how to authenticate",
or "how to use this", they are asking about YOU (the RAG Documentation Assistant),
NOT about generic FastAPI examples.

Current System Information:
- Name: RAG Documentation Assistant
- Framework: FastAPI (async)
- Authentication: Session-based (automatic, no OAuth2)
- Database: SQLite + ChromaDB
- Endpoints: /api/chat, /api/history, /api/stats, /api/feedback

When answering questions about authentication:
- This API uses SESSION-BASED authentication (cookies)
- NO OAuth2, NO JWT, NO API keys for regular users
- Sessions are created automatically on first request

Always provide specific examples for THIS system, not generic examples.
"""
```

**Entregáveis**:
1. Função `is_self_query()`
2. Filtro de metadata em `search_documents()`
3. System prompt atualizado
4. Lista de keywords expandível

**Critérios de Aceitação**:
- [ ] 95%+ de precisão na detecção
- [ ] Self-queries retornam docs internos
- [ ] Respostas específicas do sistema
- [ ] Logging de detecções

---

##### TASK 10.2.4 - Corrigir Bug de Related Concepts Truncado
**Arquivo**: `rag_engine_async.py`, `code_generator.py`
**Responsável**: @backend-architect
**Prioridade**: 🟡 Média
**Estimativa**: 1 dia

**Descrição**:
Investigar e corrigir truncamento do campo `related_concepts`.

**Investigação**:
```python
# Possíveis causas:
# 1. Serialização JSON incorreta
# 2. Truncamento de string em algum middleware
# 3. Limite de tamanho de resposta
# 4. Erro de encoding
```

**Fix**:
```python
# Em rag_engine_async.py

async def generate_response(self, query: str, conversation_history: List = None):
    # ... código existente ...

    # ANTES (possível causa do bug)
    related_concepts = self._extract_concepts(context)[:5]  # OK

    # Garantir que concepts são strings simples
    related_concepts = [str(concept).strip() for concept in related_concepts]

    # Validar que não há caracteres problemáticos
    related_concepts = [
        concept for concept in related_concepts
        if concept and len(concept) < 100  # Evitar concepts muito longos
    ]

    response_data = {
        "answer": answer,
        "sources": sources,
        "examples": examples,
        "endpoints": endpoints,
        "authentication": auth_info,
        "parameters": parameters,
        "response_format": response_format,
        "error_codes": error_codes,
        "related_concepts": related_concepts  # ← Agora validado
    }

    # Validar JSON serializável
    try:
        json.dumps(response_data)
    except TypeError as e:
        logger.error(f"JSON serialization error: {e}")
        # Remover campos problemáticos
        response_data["related_concepts"] = []

    return response_data
```

**Testes**:
```python
# test_related_concepts.py

@pytest.mark.asyncio
async def test_related_concepts_not_truncated():
    """Test that related_concepts field is complete"""
    rag_engine = AsyncRAGEngine()

    response = await rag_engine.generate_response(
        "How to create FastAPI endpoint?"
    )

    concepts = response.get("related_concepts", [])

    # Verificar que é lista válida
    assert isinstance(concepts, list)

    # Verificar que todos são strings
    assert all(isinstance(c, str) for c in concepts)

    # Verificar que nenhum está truncado
    for concept in concepts:
        assert not concept.endswith('[')  # Não truncado
        assert len(concept) < 100  # Razoável

    # Verificar serializável
    json_str = json.dumps({"related_concepts": concepts})
    assert '"related_concepts"' in json_str
```

**Entregáveis**:
1. Bug fix implementado
2. Validação de JSON serialization
3. Testes automatizados
4. Logging de erros de serialização

**Critérios de Aceitação**:
- [ ] Campo `related_concepts` sempre completo
- [ ] JSON válido 100% das vezes
- [ ] Testes passando
- [ ] Zero erros de serialização em produção

---

### **FASE 10.3 - Backend: Prompt Engineering** (Semana 2)
**Responsável**: @rag-specialist
**Prioridade**: 🟡 Média
**Duração**: 3 dias

---

##### TASK 10.3.1 - Otimizar System Prompts
**Arquivo**: `rag_engine_async.py`
**Responsável**: @rag-specialist
**Prioridade**: 🟡 Média
**Estimativa**: 2 dias

**Descrição**:
Melhorar os prompts do sistema para gerar respostas mais específicas e úteis.

**Implementação**:
```python
# Em rag_engine_async.py

ENHANCED_SYSTEM_PROMPT = """
You are an expert AI assistant specialized in API documentation and software development.

## Your Role
- Provide accurate, detailed answers about APIs, frameworks, and development tools
- Generate working code examples in multiple languages
- Cite sources for all information
- Be specific and practical

## Response Format
Always structure your responses with:
1. **Direct Answer**: Clear, concise answer to the question
2. **Code Examples**: Working code in relevant languages (Python, JavaScript, cURL, etc.)
3. **Endpoints**: List relevant API endpoints with method, path, and description
4. **Authentication**: Explain auth requirements if applicable
5. **Best Practices**: Include tips and gotchas

## Self-Awareness
When users ask about "this API", "this system", or "how to use this":
- They mean the RAG Documentation Assistant API (this system)
- Provide specific information about THIS system, not generic examples
- Mention session-based authentication (no OAuth2)
- Reference actual endpoints: /api/chat, /api/history, /api/stats

## Code Quality
- All code must be production-ready and follow best practices
- Include error handling
- Add comments for complex logic
- Use type hints (Python) and JSDoc (JavaScript)
- Provide complete, runnable examples

## Sources
- Always cite official documentation
- Include relevance score
- Provide direct links

## Tone
- Professional but friendly
- Concise but complete
- Assume intermediate technical knowledge
"""

API_SPECIFIC_PROMPT = """
## Context About This Query
Topic: {topic}
User Level: {level}
Query Type: {query_type}

## Retrieved Documentation
{context}

## Conversation History
{history}

## Instructions
Based on the above context, provide a comprehensive answer that includes:
1. Clear explanation
2. 2-3 code examples (different languages if applicable)
3. Related API endpoints
4. Authentication requirements
5. Common pitfalls to avoid

Be specific to the technologies mentioned in the documentation.
"""
```

**Few-Shot Examples**:
```python
FEW_SHOT_EXAMPLES = [
    {
        "query": "How to create a POST endpoint?",
        "good_response": {
            "answer": "To create a POST endpoint in FastAPI...",
            "examples": [
                {"language": "Python", "code": "..."},
                {"language": "cURL", "code": "..."}
            ],
            "endpoints": [{"method": "POST", "path": "...", "description": "..."}]
        }
    },
    {
        "query": "How to authenticate with this API?",
        "good_response": {
            "answer": "This API (RAG Documentation Assistant) uses session-based authentication...",
            "authentication": {
                "type": "Session-based",
                "description": "Automatic cookie-based sessions, no manual auth required"
            }
        }
    }
]
```

**Entregáveis**:
1. System prompt otimizado
2. Few-shot examples
3. Prompts específicos por tipo de query
4. Template de resposta estruturada

**Critérios de Aceitação**:
- [ ] Respostas 30%+ mais específicas
- [ ] 90%+ incluem exemplos de código
- [ ] Self-awareness 95%+ precisa
- [ ] Feedback de usuários positivo

---

##### TASK 10.3.2 - Adicionar Classificação de Queries
**Arquivo**: `rag_engine_async.py`
**Responsável**: @rag-specialist
**Prioridade**: 🟢 Baixa
**Estimativa**: 1 dia

**Descrição**:
Classificar queries em categorias para aplicar prompts especializados.

**Implementação**:
```python
class QueryClassifier:
    """Classify queries to apply specialized prompts"""

    CATEGORIES = {
        "authentication": ["auth", "login", "token", "api key", "oauth", "session"],
        "endpoints": ["endpoint", "route", "api", "request", "response"],
        "code_example": ["example", "code", "how to", "show me", "implement"],
        "error": ["error", "exception", "bug", "fix", "problem", "issue"],
        "setup": ["install", "setup", "configure", "initialize", "start"],
        "self": ["this api", "this system", "your api", "how do i use"]
    }

    def classify(self, query: str) -> List[str]:
        """Return list of categories for query"""
        query_lower = query.lower()
        categories = []

        for category, keywords in self.CATEGORIES.items():
            if any(kw in query_lower for kw in keywords):
                categories.append(category)

        return categories if categories else ["general"]

    def get_specialized_prompt(self, categories: List[str]) -> str:
        """Get prompt based on categories"""
        prompts = {
            "authentication": "Focus on authentication methods, security, and token handling.",
            "endpoints": "Provide endpoint details: method, path, parameters, response format.",
            "code_example": "Provide 2-3 working code examples in different languages.",
            "error": "Explain the error, root cause, and step-by-step solution.",
            "self": "This query is about the RAG Documentation Assistant itself. Provide specific info about THIS system."
        }

        return " ".join([prompts.get(cat, "") for cat in categories])
```

**Entregáveis**:
1. `QueryClassifier` class
2. Categoria detection
3. Specialized prompts
4. Métricas de classificação

**Critérios de Aceitação**:
- [ ] 85%+ precisão na classificação
- [ ] Prompts especializados aplicados
- [ ] Performance < 50ms
- [ ] Logs de classificação

---

### **FASE 10.4 - Testing & QA** (Semana 2-3)
**Responsável**: @qa-engineer
**Prioridade**: 🟡 Média
**Duração**: 4 dias

---

##### TASK 10.4.1 - Testes de UI/UX
**Arquivo**: `tests/test_chat_ui.py` (novo)
**Responsável**: @qa-engineer
**Prioridade**: 🟡 Média
**Estimativa**: 2 dias

**Descrição**:
Criar testes automatizados para a nova interface de chat.

**Testes**:
```python
# tests/test_chat_ui.py

import pytest
from playwright.async_api import async_playwright

@pytest.mark.asyncio
async def test_chat_rendering():
    """Test that chat renders responses correctly"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto("http://localhost:8000/chat")

        # Enviar query
        await page.fill("#user-input", "How to create FastAPI endpoint?")
        await page.click("#send-button")

        # Aguardar resposta
        await page.wait_for_selector(".ai-message", timeout=30000)

        # Verificar elementos renderizados
        assert await page.query_selector(".answer-section")
        assert await page.query_selector(".code-example")
        assert await page.query_selector(".sources-section")

        # Verificar syntax highlighting
        code_blocks = await page.query_selector_all("pre code[class*='language-']")
        assert len(code_blocks) > 0

        # Verificar links clicáveis
        source_links = await page.query_selector_all(".source-card")
        assert len(source_links) > 0

        await browser.close()

@pytest.mark.asyncio
async def test_copy_code_button():
    """Test copy button functionality"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto("http://localhost:8000/chat")
        await page.fill("#user-input", "Show Python example")
        await page.click("#send-button")

        await page.wait_for_selector(".copy-button")

        # Click copy button
        await page.click(".copy-button")

        # Verify success feedback
        assert await page.query_selector(".copy-success")

        await browser.close()

@pytest.mark.asyncio
async def test_loading_states():
    """Test loading indicators"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto("http://localhost:8000/chat")

        # Start query
        await page.fill("#user-input", "test query")
        await page.click("#send-button")

        # Verify loading indicator appears
        loading = await page.wait_for_selector(".typing-indicator", timeout=1000)
        assert loading

        # Verify loading disappears when done
        await page.wait_for_selector(".typing-indicator", state="hidden", timeout=30000)

        await browser.close()
```

**Entregáveis**:
1. 10+ testes de UI
2. Testes de acessibilidade
3. Testes de responsividade
4. Report de cobertura

**Critérios de Aceitação**:
- [ ] 100% dos componentes testados
- [ ] Testes passam em Chrome, Firefox, Safari
- [ ] Mobile/tablet testados
- [ ] Acessibilidade WCAG 2.1 AA

---

##### TASK 10.4.2 - Testes de Self-Awareness
**Arquivo**: `tests/test_self_awareness.py` (novo)
**Responsável**: @qa-engineer
**Prioridade**: 🔴 Alta
**Estimativa**: 1.5 dias

**Descrição**:
Validar que o sistema responde corretamente a perguntas sobre si mesmo.

**Testes**:
```python
# tests/test_self_awareness.py

import pytest
from httpx import AsyncClient

SELF_QUERIES = [
    "How to authenticate with this API?",
    "How do I use this system?",
    "What endpoints does this API have?",
    "How to get started with your API?",
    "What is the authentication method?",
    "How to make a request to this API?",
    "What's the rate limit?",
    "How do sessions work here?"
]

@pytest.mark.asyncio
async def test_self_query_detection():
    """Test that self-queries are detected"""
    from rag_engine_async import is_self_query

    for query in SELF_QUERIES:
        assert is_self_query(query), f"Failed to detect: {query}"

@pytest.mark.asyncio
async def test_self_query_responses():
    """Test that self-queries return internal docs"""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        for query in SELF_QUERIES:
            response = await client.post("/api/chat", json={"query": query})
            assert response.status_code == 200

            data = response.json()

            # Verificar que menciona o sistema correto
            answer = data.get("response", "").lower()
            assert "session" in answer or "cookie" in answer
            assert "oauth" not in answer  # Não deve mencionar OAuth2

            # Verificar fontes internas
            sources = data.get("sources", [])
            internal_sources = [s for s in sources if "internal://" in s.get("url", "")]
            assert len(internal_sources) > 0, f"No internal sources for: {query}"

@pytest.mark.asyncio
async def test_auth_response_accuracy():
    """Test authentication query returns correct info"""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post("/api/chat", json={
            "query": "How to authenticate with this API?"
        })

        data = response.json()
        answer = data.get("response", "").lower()

        # DEVE mencionar:
        assert "session" in answer
        assert "cookie" in answer or "automatic" in answer

        # NÃO DEVE mencionar:
        assert "oauth2" not in answer
        assert "jwt" not in answer
        assert "bearer token" not in answer

        # Verificar auth info estruturada
        auth = data.get("authentication", {})
        assert "session" in auth.get("type", "").lower()
```

**Entregáveis**:
1. 15+ test cases de self-awareness
2. Validação de respostas
3. Métricas de precisão
4. Report de falhas

**Critérios de Aceitação**:
- [ ] 95%+ queries detectadas corretamente
- [ ] 100% respostas sem mencionar OAuth2
- [ ] Fontes internas sempre incluídas
- [ ] Zero false positives

---

##### TASK 10.4.3 - Performance Testing
**Arquivo**: `tests/test_performance.py`
**Responsável**: @qa-engineer
**Prioridade**: 🟡 Média
**Estimativa**: 0.5 dia

**Descrição**:
Validar que as melhorias não degradaram performance.

**Benchmarks**:
```python
# tests/test_performance.py

import pytest
import asyncio
from httpx import AsyncClient
import time

@pytest.mark.asyncio
async def test_response_time():
    """Test that responses are fast enough"""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        queries = [
            "What is FastAPI?",
            "How to create endpoint?",
            "Python async example",
        ]

        for query in queries:
            start = time.time()
            response = await client.post("/api/chat", json={"query": query})
            elapsed = time.time() - start

            assert response.status_code == 200
            assert elapsed < 15.0, f"Query took {elapsed}s (limit: 15s)"

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test system handles concurrent requests"""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        async def make_request(i):
            response = await client.post("/api/chat", json={
                "query": f"Test query {i}"
            })
            return response.status_code == 200

        # 10 concurrent requests
        results = await asyncio.gather(*[make_request(i) for i in range(10)])

        assert all(results), "Some concurrent requests failed"

@pytest.mark.asyncio
async def test_frontend_render_time():
    """Test frontend rendering is fast"""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto("http://localhost:8000/chat")
        await page.fill("#user-input", "test")

        start = time.time()
        await page.click("#send-button")
        await page.wait_for_selector(".ai-message")
        render_time = time.time() - start

        # Rendering should be instant (excluding API call)
        # Measure just the render after response arrives
        assert render_time < 0.5, f"Rendering took {render_time}s"

        await browser.close()
```

**Métricas Alvo**:
- Response time: < 15s para queries complexas
- Frontend render: < 500ms
- Concurrent requests: 10+ simultâneos sem falhas
- Time to interactive: < 2s

**Entregáveis**:
1. Suite de benchmarks
2. Performance report
3. Comparação antes/depois
4. Identificação de gargalos

**Critérios de Aceitação**:
- [ ] Todas as métricas dentro do alvo
- [ ] Sem regressão de performance
- [ ] Report documentado
- [ ] CI/CD integrado

---

### **FASE 10.5 - Security & DevOps** (Semana 3)
**Responsável**: @security-specialist + @devops-engineer
**Prioridade**: 🟢 Baixa
**Duração**: 2 dias

---

##### TASK 10.5.1 - Security Audit de Novas Features
**Arquivo**: `SECURITY_AUDIT_REPORT.md` (novo)
**Responsável**: @security-specialist
**Prioridade**: 🟡 Média
**Estimativa**: 1 dia

**Checklist de Segurança**:
- [ ] XSS prevention em renderização HTML
- [ ] CSRF tokens em forms (se aplicável)
- [ ] Rate limiting funcionando
- [ ] Input validation em todos campos
- [ ] Sanitização de output
- [ ] SQL injection protection
- [ ] Secrets não expostos no frontend
- [ ] HTTPS enforced
- [ ] Security headers corretos
- [ ] Dependency vulnerabilities (npm audit, safety)

**Entregáveis**:
1. Security audit report
2. Lista de vulnerabilidades encontradas
3. Recomendações de fixes
4. Plano de remediação

**Critérios de Aceitação**:
- [ ] Zero vulnerabilidades críticas
- [ ] Todas médias/altas corrigidas
- [ ] Report documentado
- [ ] Aprovação do security specialist

---

##### TASK 10.5.2 - Deployment & Monitoring
**Arquivo**: `deploy/phase10_deployment.md` (novo)
**Responsável**: @devops-engineer
**Prioridade**: 🟡 Média
**Estimativa**: 1 dia

**Deployment Plan**:
```markdown
# Phase 10 Deployment Plan

## Pre-Deployment
1. Run all tests (UI, backend, integration)
2. Performance benchmarks
3. Security scan
4. Backup database
5. Backup ChromaDB

## Deployment Steps
1. Deploy to staging
2. Smoke tests em staging
3. Load testing em staging
4. User acceptance testing (2-3 users)
5. Deploy to production (blue-green)
6. Monitor metrics por 24h

## Rollback Plan
If any issue:
1. Switch traffic to old version (blue-green)
2. Investigate issue
3. Fix and redeploy

## Monitoring
- Response time (p50, p95, p99)
- Error rate
- Self-query detection accuracy
- Frontend render time
- User satisfaction (feedback)
```

**Métricas de Monitoramento**:
```python
# monitoring/phase10_metrics.py

METRICS = {
    "chat_response_time_p95": {"target": 15, "unit": "seconds"},
    "frontend_render_time_p95": {"target": 0.5, "unit": "seconds"},
    "self_query_accuracy": {"target": 0.95, "unit": "ratio"},
    "error_rate": {"target": 0.01, "unit": "ratio"},
    "user_satisfaction": {"target": 0.85, "unit": "ratio"}
}
```

**Entregáveis**:
1. Deployment plan
2. Rollback procedure
3. Monitoring dashboard
4. Alert rules

**Critérios de Aceitação**:
- [ ] Deploy bem-sucedido sem downtime
- [ ] Todas métricas dentro do alvo
- [ ] Monitoring funcionando
- [ ] Rollback testado

---

## 📊 Delegação de Tasks por Agent

### @frontend-developer (5 tasks, 5 dias)
1. ✅ TASK 10.1.1 - Refatorar Renderização de Respostas (2d)
2. ✅ TASK 10.1.2 - Syntax Highlighting Avançado (1d)
3. ✅ TASK 10.1.3 - UI Components (1.5d)
4. ✅ TASK 10.1.4 - Loading States e Feedback (0.5d)

**Total**: 5 dias de trabalho

---

### @rag-specialist (4 tasks, 6 dias)
1. ✅ TASK 10.2.1 - Criar Documentação Interna (2d)
2. ✅ TASK 10.2.2 - Indexar Docs Internas (1d)
3. ✅ TASK 10.3.1 - Otimizar System Prompts (2d)
4. ✅ TASK 10.3.2 - Classificação de Queries (1d)

**Total**: 6 dias de trabalho

---

### @backend-architect (2 tasks, 2.5 dias)
1. ✅ TASK 10.2.3 - Detecção de Self-Queries (1.5d)
2. ✅ TASK 10.2.4 - Fix Related Concepts Bug (1d)

**Total**: 2.5 dias de trabalho

---

### @qa-engineer (3 tasks, 4 dias)
1. ✅ TASK 10.4.1 - Testes de UI/UX (2d)
2. ✅ TASK 10.4.2 - Testes de Self-Awareness (1.5d)
3. ✅ TASK 10.4.3 - Performance Testing (0.5d)

**Total**: 4 dias de trabalho

---

### @security-specialist (1 task, 1 dia)
1. ✅ TASK 10.5.1 - Security Audit (1d)

**Total**: 1 dia de trabalho

---

### @devops-engineer (1 task, 1 dia)
1. ✅ TASK 10.5.2 - Deployment & Monitoring (1d)

**Total**: 1 dia de trabalho

---

## 📅 Timeline

```
Semana 1 (Dias 1-5):
├─ Frontend: Tasks 10.1.1, 10.1.2, 10.1.3, 10.1.4 (paralelo)
├─ RAG Specialist: Tasks 10.2.1, 10.2.2 (paralelo)
└─ Backend Architect: Task 10.2.3 (paralelo)

Semana 2 (Dias 6-10):
├─ Backend Architect: Task 10.2.4
├─ RAG Specialist: Tasks 10.3.1, 10.3.2
└─ QA Engineer: Tasks 10.4.1, 10.4.2

Semana 3 (Dias 11-15):
├─ QA Engineer: Task 10.4.3
├─ Security: Task 10.5.1
└─ DevOps: Task 10.5.2
```

---

## ✅ Checklist de Conclusão

### Frontend
- [ ] Todas as respostas renderizadas como UI rica
- [ ] Syntax highlighting funcionando em todas linguagens
- [ ] Fontes clicáveis
- [ ] Loading states implementados
- [ ] Animações suaves
- [ ] Responsivo (mobile/tablet/desktop)
- [ ] Acessível (WCAG 2.1 AA)

### Backend
- [ ] Documentação interna indexada no ChromaDB
- [ ] Self-queries detectadas com 95%+ precisão
- [ ] Respostas específicas do sistema
- [ ] Bug de related_concepts corrigido
- [ ] System prompts otimizados
- [ ] Query classification funcionando

### Testing
- [ ] 100% dos testes de UI passando
- [ ] 95%+ precisão em testes de self-awareness
- [ ] Performance dentro das métricas alvo
- [ ] Security audit aprovado
- [ ] Load testing bem-sucedido

### DevOps
- [ ] Deploy em produção sem downtime
- [ ] Monitoring funcionando
- [ ] Alerts configurados
- [ ] Rollback testado
- [ ] Documentação atualizada

---

## 🎯 Métricas de Sucesso Final

| Métrica | Antes | Meta | Medição |
|---------|-------|------|---------|
| Syntax highlighting | 0% | 100% | % de blocos com highlight |
| Fontes clicáveis | 0% | 100% | % de fontes com links |
| Self-query accuracy | ~30% | 95% | True positives / Total |
| Response time (p95) | ~15s | <15s | 95th percentile |
| Frontend render | ~1s | <0.5s | Time to paint |
| User satisfaction | ~60% | 85% | Feedback positivo |
| Error rate | ~5% | <1% | Erros / Total requests |

---

## 📝 Notas Finais

### Dependências
- Frontend tasks podem começar imediatamente
- Backend tasks 10.2.3 e 10.2.1/10.2.2 são paralelas
- QA tasks dependem de conclusão do frontend e backend
- DevOps task depende de QA approval

### Riscos
1. **Risco Alto**: Self-query detection pode ter falsos positivos → Mitigação: Testes extensivos
2. **Risco Médio**: Performance pode degradar com UI rica → Mitigação: Lazy loading, virtualização
3. **Risco Baixo**: Syntax highlighting pode falhar em linguagens raras → Mitigação: Fallback para plain text

### Aprovações Necessárias
- [ ] Product Owner - Aprovação do plano
- [ ] Tech Lead - Aprovação técnica
- [ ] Security Lead - Aprovação de segurança
- [ ] DevOps Lead - Aprovação de deployment

---

**Plano criado em**: 2025-12-19
**Última atualização**: 2025-12-19
**Status**: ✅ Pronto para execução
**Próxima revisão**: Após Semana 1
