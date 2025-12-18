# DEV_3: LangSmith Observability

**Desenvolvedor:** DEV_3
**Fase:** 9C - LangSmith Integration
**Prioridade:** ⭐⭐ ALTA
**Estimativa:** 2 horas
**Dependências:** DEV_1 (agent precisa estar funcionando)

---

## 🎯 Objetivo

Implementar **observability completo** com LangSmith para tracing, debugging e monitoring de todas as chamadas LLM.

---

## 📦 Entregas

1. **`langsmith_config.py`** - LangSmith setup e configuração
2. **`callbacks/tracing.py`** - Custom callbacks
3. **`monitoring/metrics.py`** - Metrics aggregation
4. **`test_observability.py`** - Tests

---

## 📝 Implementação

### 1. Setup LangSmith (Grátis)

```bash
# 1. Criar conta: https://smith.langchain.com
# 2. Criar projeto: "rag-documentation-assistant"
# 3. Copiar API key

# 4. Adicionar ao .env:
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=rag-documentation-assistant
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### 2. `langsmith_config.py`

```python
import os
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from config import Config

class LangSmithConfig:
    """LangSmith observability configuration"""

    @staticmethod
    def is_enabled() -> bool:
        """Check if LangSmith is enabled"""
        return os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    @staticmethod
    def get_client() -> Client:
        """Get LangSmith client"""
        if not LangSmithConfig.is_enabled():
            return None

        return Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        )

    @staticmethod
    def get_tracer(run_name: str = "default") -> LangChainTracer:
        """Get LangChain tracer for callbacks"""
        if not LangSmithConfig.is_enabled():
            return None

        return LangChainTracer(
            project_name=os.getenv("LANGSMITH_PROJECT", "rag-assistant"),
            client=LangSmithConfig.get_client()
        )

    @staticmethod
    def get_callbacks(run_name: str = "agent_run"):
        """Get callback list for agent/chain"""
        if not LangSmithConfig.is_enabled():
            return []

        tracer = LangSmithConfig.get_tracer(run_name)
        return [tracer] if tracer else []
```

### 3. Integrar com Agent (modificar DEV_1's code)

```python
# Em langchain_agent.py, modificar:

from langsmith_config import LangSmithConfig

class DocumentationAgent:
    def __init__(self):
        # ... existing code ...

        # Add LangSmith callbacks
        self.callbacks = LangSmithConfig.get_callbacks("documentation_agent")

    async def arun(self, query: str, conversation_history=None):
        # Add callbacks to execution
        result = await self.agent_executor.ainvoke(
            input_data,
            config={"callbacks": self.callbacks}  # ← ADD THIS
        )
```

### 4. `callbacks/tracing.py` - Custom Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict
import time
import logging

logger = logging.getLogger(__name__)

class MetricsCallbackHandler(BaseCallbackHandler):
    """Custom callback for collecting metrics"""

    def __init__(self):
        self.llm_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs):
        """Track LLM call start"""
        self.llm_calls += 1
        self.start_time = time.time()
        logger.info(f"LLM call #{self.llm_calls} started")

    def on_llm_end(self, response, **kwargs):
        """Track LLM call end"""
        elapsed = time.time() - self.start_time
        tokens = response.llm_output.get('token_usage', {})

        self.total_tokens += tokens.get('total_tokens', 0)

        # Estimate cost (GPT-4: $0.03/1K input, $0.06/1K output)
        input_tokens = tokens.get('prompt_tokens', 0)
        output_tokens = tokens.get('completion_tokens', 0)
        cost = (input_tokens * 0.03 + output_tokens * 0.06) / 1000
        self.total_cost += cost

        logger.info(
            f"LLM call completed: {elapsed:.2f}s, "
            f"Tokens: {tokens.get('total_tokens', 0)}, "
            f"Cost: ${cost:.4f}"
        )

    def get_metrics(self) -> Dict:
        """Get collected metrics"""
        return {
            'llm_calls': self.llm_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }
```

### 5. `monitoring/metrics.py`

```python
from typing import Dict
from langsmith import Client
from datetime import datetime, timedelta

class MetricsCollector:
    """Collect and aggregate metrics from LangSmith"""

    def __init__(self):
        from langsmith_config import LangSmithConfig
        self.client = LangSmithConfig.get_client()

    async def get_daily_metrics(self) -> Dict:
        """Get metrics for last 24 hours"""
        if not self.client:
            return {'error': 'LangSmith not enabled'}

        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

        # Get runs from last 24h
        runs = list(self.client.list_runs(
            project_name=os.getenv("LANGSMITH_PROJECT"),
            start_time=start_time,
            end_time=end_time
        ))

        # Aggregate metrics
        total_runs = len(runs)
        successful = sum(1 for r in runs if not r.error)
        failed = total_runs - successful

        total_tokens = sum(
            r.outputs.get('token_usage', {}).get('total_tokens', 0)
            for r in runs if r.outputs
        )

        avg_latency = sum(
            (r.end_time - r.start_time).total_seconds()
            for r in runs if r.end_time and r.start_time
        ) / total_runs if total_runs > 0 else 0

        return {
            'period': '24h',
            'total_runs': total_runs,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total_runs * 100) if total_runs > 0 else 0,
            'total_tokens': total_tokens,
            'avg_latency': avg_latency,
            'estimated_cost': total_tokens * 0.04 / 1000  # Rough estimate
        }
```

### 6. Add Endpoint para Metrics

```python
# Em routes_agent.py, adicionar:

from monitoring.metrics import MetricsCollector

@router.get("/metrics")
async def get_metrics():
    """Get LangSmith metrics"""
    collector = MetricsCollector()
    return await collector.get_daily_metrics()
```

---

## ✅ Critérios de Aceitação

- [ ] LangSmith account criado e configurado
- [ ] Tracing funcionando em todas LLM calls
- [ ] Metrics dashboard acessível em smith.langchain.com
- [ ] Custom callbacks coletando métricas
- [ ] Endpoint `/api/agent/metrics` retornando dados
- [ ] Custo por query sendo tracked
- [ ] Latency P95 < 2s
- [ ] Zero impacto em performance

---

## 🧪 Como Testar

```bash
# 1. Fazer query via agent
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI?"}'

# 2. Ver trace no LangSmith
# → Abrir https://smith.langchain.com
# → Ver projeto "rag-documentation-assistant"
# → Ver trace detalhado da query

# 3. Ver métricas
curl http://localhost:8000/api/agent/metrics
```

---

**🎯 Foco:** Observability é crítico para produção. Capture TUDO!
