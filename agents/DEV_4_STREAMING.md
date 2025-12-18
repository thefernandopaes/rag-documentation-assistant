# DEV_4: Streaming Responses

**Desenvolvedor:** DEV_4
**Fase:** 9D - Streaming Responses
**Prioridade:** ⭐ MÉDIA
**Estimativa:** 2 horas
**Dependências:** DEV_1 (agent precisa estar funcionando)

---

## 🎯 Objetivo

Implementar **streaming de respostas** em tempo real usando Server-Sent Events (SSE) para melhor UX.

---

## 📦 Entregas

1. **`streaming/stream_handler.py`** - SSE handler
2. **`routes_streaming.py`** - Streaming endpoints
3. **`frontend_demo.html`** - Demo (opcional)
4. **`test_streaming.py`** - Tests

---

## 📝 Implementação

### 1. `streaming/stream_handler.py`

```python
from typing import AsyncGenerator
import asyncio
import json
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult

class StreamingCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming agent responses"""

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.done = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when new token is generated"""
        asyncio.create_task(self.queue.put(token))

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes"""
        self.done = True
        asyncio.create_task(self.queue.put(None))  # Sentinel

    async def aiter(self) -> AsyncGenerator[str, None]:
        """Async iterator for tokens"""
        while not self.done or not self.queue.empty():
            token = await self.queue.get()
            if token is None:
                break
            yield token


async def stream_agent_response(agent, query: str) -> AsyncGenerator[str, None]:
    """
    Stream agent response token by token.

    Args:
        agent: DocumentationAgent instance
        query: User query

    Yields:
        Token strings as they're generated
    """
    # Create streaming callback
    callback = StreamingCallbackHandler()

    # Modify agent to use streaming
    agent.llm.streaming = True
    agent.llm.callbacks = [callback]

    # Start agent execution in background
    task = asyncio.create_task(agent.arun(query))

    # Stream tokens as they arrive
    async for token in callback.aiter():
        yield f"data: {json.dumps({'token': token})}\n\n"

    # Wait for agent to complete
    await task

    # Send final event
    yield f"data: {json.dumps({'done': True})}\n\n"
```

### 2. `routes_streaming.py`

```python
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json

from langchain_agent import DocumentationAgent
from streaming.stream_handler import stream_agent_response
from schemas import ChatRequest
from dependencies import get_session_id, validate_rate_limit

router = APIRouter(prefix="/api/stream", tags=["Streaming"])

def get_agent():
    """Get agent instance"""
    from routes_agent import get_agent
    return get_agent()


@router.post("/chat")
async def stream_chat(
    request: ChatRequest,
    session_id: str = Depends(get_session_id),
    agent: DocumentationAgent = Depends(get_agent),
    _rate_limit: None = Depends(validate_rate_limit)
):
    """
    Stream agent response in real-time.

    Returns Server-Sent Events (SSE) stream.

    Usage:
        const eventSource = new EventSource('/api/stream/chat');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.token) {
                console.log(data.token);
            } else if (data.done) {
                eventSource.close();
            }
        };
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events"""
        try:
            async for event in stream_agent_response(agent, request.query):
                yield event

        except Exception as e:
            error_event = f"data: {json.dumps({'error': str(e)})}\n\n"
            yield error_event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
```

### 3. `frontend_demo.html` (Opcional - para demonstração)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Streaming Demo</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; }
        #response { border: 1px solid #ccc; padding: 20px; min-height: 200px; }
        #query { width: 100%; padding: 10px; margin: 10px 0; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .token { display: inline; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>Agent Streaming Demo</h1>

    <input type="text" id="query" placeholder="Enter your query..." value="What is FastAPI?">
    <button onclick="streamResponse()">Send Query</button>

    <h2>Response:</h2>
    <div id="response"></div>

    <script>
        let eventSource = null;

        function streamResponse() {
            const query = document.getElementById('query').value;
            const responseDiv = document.getElementById('response');

            // Clear previous response
            responseDiv.innerHTML = '<span class="loading">Thinking...</span>';

            // Close existing stream
            if (eventSource) {
                eventSource.close();
            }

            // Create new EventSource
            eventSource = new EventSource('/api/stream/chat?' + new URLSearchParams({
                query: query
            }));

            // Clear loading message on first token
            let firstToken = true;

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (firstToken) {
                    responseDiv.innerHTML = '';
                    firstToken = false;
                }

                if (data.token) {
                    // Append token
                    const span = document.createElement('span');
                    span.className = 'token';
                    span.textContent = data.token;
                    responseDiv.appendChild(span);
                } else if (data.done) {
                    // Stream complete
                    eventSource.close();
                    console.log('Stream complete');
                } else if (data.error) {
                    // Error occurred
                    responseDiv.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
                    eventSource.close();
                }
            };

            eventSource.onerror = (error) => {
                console.error('EventSource error:', error);
                responseDiv.innerHTML += '<br><span style="color: red;">Connection error</span>';
                eventSource.close();
            };
        }
    </script>
</body>
</html>
```

### 4. `test_streaming.py`

```python
import pytest
from httpx import AsyncClient
import json

@pytest.mark.asyncio
async def test_streaming_endpoint(async_client: AsyncClient):
    """Test streaming endpoint returns SSE"""

    async with async_client.stream(
        "POST",
        "/api/stream/chat",
        json={"query": "What is FastAPI?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"

        # Read first few events
        tokens = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if 'token' in data:
                    tokens.append(data['token'])
                elif data.get('done'):
                    break

        # Should have received some tokens
        assert len(tokens) > 0
```

---

## ✅ Critérios de Aceitação

- [ ] Endpoint `/api/stream/chat` funcionando
- [ ] SSE stream retornando tokens em tempo real
- [ ] Frontend demo funcionando (se implementado)
- [ ] Latency < 500ms para primeiro token
- [ ] Handling de erros correto
- [ ] Graceful shutdown do stream
- [ ] Tests passando

---

## 🧪 Como Testar

```bash
# 1. Testar com curl (exibe tokens conforme chegam)
curl -N -X POST http://localhost:8000/api/stream/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI?"}' \
  | while IFS= read -r line; do echo "$line"; done

# 2. Abrir demo frontend
# → Abrir http://localhost:8000/frontend_demo.html
# → Testar query
# → Ver tokens aparecendo em tempo real

# 3. Teste de performance (primeiro token)
time curl -N -X POST http://localhost:8000/api/stream/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello"}' \
  | head -1
# → Deve ser < 500ms
```

---

**💡 Dica:** SSE é perfeito para streaming unidirecional. Para chat bi-direcional, considere WebSockets no futuro.
