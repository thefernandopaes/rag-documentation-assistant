# DEV_1: LangChain Agents & Custom Tools

**Desenvolvedor:** DEV_1
**Fase:** 9A - Agents & Custom Tools
**Prioridade:** ⭐⭐⭐ CRÍTICA
**Estimativa:** 3-4 horas
**Dependências:** Nenhuma (pode começar imediatamente)

---

## 🎯 Objetivo

Implementar um **ReAct Agent do LangChain** que orquestra múltiplas tools customizadas para responder queries complexas sobre documentação de APIs.

### O que você vai construir:

```
User Query → Agent → Tool Selection → Tool Execution → Response
                ↓
          [RAG Tool, Code Gen Tool, Validator Tool]
```

---

## 📦 Entregas

### Arquivos a criar:

1. **`langchain_agent.py`** - Core agent logic
2. **`tools/rag_tool.py`** - RAG search tool
3. **`tools/code_generator_tool.py`** - Code generation tool
4. **`tools/validator_tool.py`** - Code validation tool
5. **`routes_agent.py`** - FastAPI endpoints
6. **`test_agent.py`** - Agent tests
7. **`schemas_agent.py`** - Pydantic models

### Arquivos a modificar:

- `fastapi_app.py` - Registrar novo router
- `pyproject.toml` - Adicionar dependências (se necessário)

---

## 🏗️ Arquitetura

```python
┌─────────────────────────────────────────┐
│          FastAPI Routes                 │
│  POST /api/agent/chat                   │
│  POST /api/agent/chat/stream (future)   │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│       LangChain Agent                   │
│  ┌──────────────────────────────────┐   │
│  │  ReAct Agent (think → act → obs) │   │
│  └──────────┬───────────────────────┘   │
│             │                            │
│  ┌──────────▼────────────────────────┐  │
│  │        Tool Selection              │  │
│  │  (Agent decides which tool to use) │  │
│  └──────────┬────────────────────────┘  │
└─────────────┼─────────────────────────┘
              │
┌─────────────▼─────────────────────────┐
│         Custom Tools                  │
│  ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │ RAG Tool │ │ Code Gen │ │Validtr│ │
│  └────┬─────┘ └────┬─────┘ └───┬───┘ │
└───────┼────────────┼─────────────┼─────┘
        │            │             │
        ▼            ▼             ▼
  [AsyncRAG]   [CodeGen]     [Validator]
  (existing)   (existing)       (new)
```

---

## 📝 Especificação Detalhada

### 1. `langchain_agent.py`

#### Imports necessários:
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import logging

from tools.rag_tool import RAGSearchTool
from tools.code_generator_tool import CodeGeneratorTool
from tools.validator_tool import CodeValidatorTool
from config import Config
```

#### Classe principal:
```python
class DocumentationAgent:
    """
    LangChain ReAct agent for API documentation queries.

    Capabilities:
    - Search documentation using RAG
    - Generate code examples
    - Validate generated code
    - Multi-step reasoning
    """

    def __init__(self):
        """Initialize agent with tools and LLM"""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",  # ou gpt-3.5-turbo para economizar
            temperature=0.2,
            api_key=Config.OPENAI_API_KEY,
            timeout=30
        )

        # Initialize tools
        self.tools = [
            RAGSearchTool(),
            CodeGeneratorTool(),
            CodeValidatorTool()
        ]

        # Create agent with ReAct prompt
        self.agent = self._create_agent()

        # Create executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,  # Para debugging
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def _create_agent(self):
        """Create ReAct agent with custom prompt"""
        template = """
        You are an expert API documentation assistant.

        You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Question: {input}
        Thought: {agent_scratchpad}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{t.name}: {t.description}" for t in self.tools]),
                "tool_names": ", ".join([t.name for t in self.tools])
            }
        )

        return create_react_agent(self.llm, self.tools, prompt)

    async def arun(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Run agent on query (async).

        Args:
            query: User query
            conversation_history: Previous messages (optional)

        Returns:
            {
                'output': str,
                'intermediate_steps': List[tuple],
                'tool_calls': List[Dict]
            }
        """
        try:
            # Prepare input with history if available
            input_data = {
                "input": query
            }

            if conversation_history:
                input_data["chat_history"] = conversation_history

            # Run agent
            result = await self.agent_executor.ainvoke(input_data)

            # Format response
            return {
                'output': result['output'],
                'intermediate_steps': result.get('intermediate_steps', []),
                'tool_calls': self._extract_tool_calls(result.get('intermediate_steps', []))
            }

        except Exception as e:
            logger.error(f"Agent execution error: {e}", exc_info=True)
            raise

    def _extract_tool_calls(self, steps: List[tuple]) -> List[Dict]:
        """Extract tool calls from intermediate steps"""
        tool_calls = []
        for action, observation in steps:
            tool_calls.append({
                'tool': action.tool,
                'input': action.tool_input,
                'output': str(observation)[:500]  # Truncate
            })
        return tool_calls
```

---

### 2. `tools/rag_tool.py`

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import asyncio

from rag_engine_async import AsyncRAGEngine

class RAGSearchInput(BaseModel):
    """Input for RAG search tool"""
    query: str = Field(description="The search query for documentation")

class RAGSearchTool(BaseTool):
    """
    Tool for searching API documentation using RAG.

    This tool searches through the vector database to find
    relevant documentation sections.
    """

    name: str = "documentation_search"
    description: str = (
        "Searches API documentation for relevant information. "
        "Use this when you need to find information about APIs, "
        "endpoints, parameters, or usage examples."
    )
    args_schema: Type[BaseModel] = RAGSearchInput

    def __init__(self):
        super().__init__()
        self.rag_engine = None

    def _get_rag_engine(self):
        """Lazy load RAG engine"""
        if self.rag_engine is None:
            self.rag_engine = AsyncRAGEngine()
        return self.rag_engine

    def _run(self, query: str) -> str:
        """Sync version (not used, but required by BaseTool)"""
        raise NotImplementedError("Use async version")

    async def _arun(self, query: str) -> str:
        """
        Execute RAG search (async).

        Args:
            query: Search query

        Returns:
            Formatted search results
        """
        try:
            engine = self._get_rag_engine()

            # Search documents
            results = await engine.search_documents(query, n_results=5)

            if not results or 'documents' not in results:
                return "No relevant documentation found."

            # Format results
            formatted = "Found relevant documentation:\n\n"
            for i, (doc, metadata) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0]
            ), 1):
                formatted += f"[{i}] {metadata.get('title', 'Unknown')}\n"
                formatted += f"{doc[:300]}...\n\n"

            return formatted

        except Exception as e:
            return f"Error searching documentation: {str(e)}"
```

---

### 3. `tools/code_generator_tool.py`

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

from code_generator import CodeExampleGenerator

class CodeGenInput(BaseModel):
    """Input for code generation tool"""
    specification: str = Field(
        description="Specification for what code to generate (language, functionality, etc.)"
    )

class CodeGeneratorTool(BaseTool):
    """
    Tool for generating code examples.

    Generates code examples based on API documentation and specifications.
    """

    name: str = "code_generator"
    description: str = (
        "Generates code examples in various languages (Python, JavaScript, cURL, etc.). "
        "Use this when the user asks for code examples or implementations."
    )
    args_schema: Type[BaseModel] = CodeGenInput

    def __init__(self):
        super().__init__()
        self.generator = CodeExampleGenerator()

    def _run(self, specification: str) -> str:
        """Sync version"""
        return self.generator.generate_example(specification)

    async def _arun(self, specification: str) -> str:
        """Async version"""
        import asyncio
        return await asyncio.to_thread(self._run, specification)
```

---

### 4. `tools/validator_tool.py`

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import ast
import subprocess

class CodeValidatorInput(BaseModel):
    """Input for code validator tool"""
    code: str = Field(description="Code to validate")
    language: str = Field(description="Programming language (python, javascript, etc.)")

class CodeValidatorTool(BaseTool):
    """
    Tool for validating generated code.

    Checks code for syntax errors and basic issues.
    """

    name: str = "code_validator"
    description: str = (
        "Validates code for syntax errors and basic issues. "
        "Use this after generating code to ensure it's correct."
    )
    args_schema: Type[BaseModel] = CodeValidatorInput

    def _run(self, code: str, language: str) -> str:
        """Validate code"""
        if language.lower() == "python":
            return self._validate_python(code)
        elif language.lower() in ["javascript", "js"]:
            return self._validate_javascript(code)
        else:
            return f"Validation not implemented for {language}"

    async def _arun(self, code: str, language: str) -> str:
        """Async version"""
        import asyncio
        return await asyncio.to_thread(self._run, code, language)

    def _validate_python(self, code: str) -> str:
        """Validate Python code"""
        try:
            ast.parse(code)
            return "✓ Python code is syntactically valid"
        except SyntaxError as e:
            return f"✗ Syntax error: {e.msg} at line {e.lineno}"

    def _validate_javascript(self, code: str) -> str:
        """Validate JavaScript code (basic check)"""
        # Simplified - would need node/eslint for real validation
        if code.count('{') != code.count('}'):
            return "✗ Unbalanced braces"
        if code.count('(') != code.count(')'):
            return "✗ Unbalanced parentheses"
        return "✓ JavaScript code looks valid (basic check)"
```

---

### 5. `routes_agent.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from langchain_agent import DocumentationAgent
from schemas_agent import AgentChatRequest, AgentChatResponse
from database_async import get_async_db
from dependencies import validate_rate_limit, get_session_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent"])

# Global agent instance (lazy loaded)
_agent_instance = None

def get_agent():
    """Get or create agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DocumentationAgent()
    return _agent_instance


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    session_id: str = Depends(get_session_id),
    agent: DocumentationAgent = Depends(get_agent),
    db: AsyncSession = Depends(get_async_db),
    _rate_limit: None = Depends(validate_rate_limit)
):
    """
    Chat with AI agent (multi-tool orchestration).

    The agent can:
    - Search documentation
    - Generate code examples
    - Validate code
    - Reason through complex queries
    """
    try:
        # Get conversation history if needed
        conversation_history = None
        if request.include_history:
            # TODO: Fetch from database
            conversation_history = []

        # Run agent
        result = await agent.arun(
            query=request.query,
            conversation_history=conversation_history
        )

        # Save to database
        # TODO: Save conversation with agent metadata

        return AgentChatResponse(
            response=result['output'],
            tool_calls=result['tool_calls'],
            intermediate_steps_count=len(result['intermediate_steps']),
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Agent chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}"
        )


@router.get("/tools")
async def list_tools(agent: DocumentationAgent = Depends(get_agent)):
    """List available agent tools"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in agent.tools
        ]
    }
```

---

### 6. `schemas_agent.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AgentChatRequest(BaseModel):
    """Request for agent chat"""
    query: str = Field(..., min_length=3, max_length=1000)
    include_history: bool = Field(default=False)
    max_iterations: int = Field(default=10, ge=1, le=20)

class ToolCall(BaseModel):
    """Single tool call information"""
    tool: str
    input: str
    output: str

class AgentChatResponse(BaseModel):
    """Response from agent chat"""
    response: str
    tool_calls: List[ToolCall]
    intermediate_steps_count: int
    session_id: str
```

---

### 7. `test_agent.py`

```python
import pytest
from langchain_agent import DocumentationAgent
from tools.rag_tool import RAGSearchTool
from tools.code_generator_tool import CodeGeneratorTool
from tools.validator_tool import CodeValidatorTool


def test_agent_initialization():
    """Test agent initializes correctly"""
    agent = DocumentationAgent()
    assert agent is not None
    assert len(agent.tools) == 3


@pytest.mark.asyncio
async def test_rag_tool():
    """Test RAG tool works"""
    tool = RAGSearchTool()
    result = await tool._arun("What is FastAPI?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_code_generator_tool():
    """Test code generator tool"""
    tool = CodeGeneratorTool()
    result = tool._run("Generate a Python function to add two numbers")
    assert isinstance(result, str)
    assert "def" in result


def test_validator_tool_python():
    """Test Python code validation"""
    tool = CodeValidatorTool()

    # Valid code
    result = tool._run("def add(a, b): return a + b", "python")
    assert "valid" in result.lower()

    # Invalid code
    result = tool._run("def add(a, b: return a + b", "python")
    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_agent_simple_query():
    """Test agent with simple query"""
    agent = DocumentationAgent()
    result = await agent.arun("What is FastAPI?")

    assert 'output' in result
    assert isinstance(result['output'], str)
    assert len(result['output']) > 0


@pytest.mark.asyncio
async def test_agent_code_generation_query():
    """Test agent with code generation request"""
    agent = DocumentationAgent()
    result = await agent.arun(
        "Generate a Python example of creating a FastAPI endpoint"
    )

    assert 'output' in result
    assert 'tool_calls' in result
    # Should have used code_generator tool
    tool_names = [tc['tool'] for tc in result['tool_calls']]
    assert 'code_generator' in tool_names
```

---

## ✅ Critérios de Aceitação

### Funcional:
- [ ] Agent consegue processar queries simples
- [ ] Agent consegue usar RAG tool para buscar docs
- [ ] Agent consegue usar code generator para gerar código
- [ ] Agent consegue usar validator para validar código
- [ ] Agent consegue combinar múltiplas tools em queries complexas
- [ ] Endpoint `/api/agent/chat` funcionando
- [ ] Endpoint `/api/agent/tools` listando tools

### Qualidade:
- [ ] Type hints completos
- [ ] Docstrings em todas classes/métodos
- [ ] Logging apropriado
- [ ] Error handling robusto

### Testes:
- [ ] Todos os testes passando
- [ ] Coverage > 80% nas tools
- [ ] Testes de integração com agent

### Performance:
- [ ] Agent responde em < 30s (queries complexas)
- [ ] RAG tool < 2s
- [ ] Code generator < 5s
- [ ] Validator < 1s

---

## 🧪 Como Testar

```bash
# 1. Instalar dependências
pip install langchain langchain-openai

# 2. Configurar env vars
export OPENAI_API_KEY=your-key

# 3. Rodar testes
pytest test_agent.py -v

# 4. Testar manualmente
python -c "
from langchain_agent import DocumentationAgent
import asyncio

agent = DocumentationAgent()
result = asyncio.run(agent.arun('What is FastAPI?'))
print(result)
"

# 5. Testar via API
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Generate a Python FastAPI hello world example"}'
```

---

## 📚 Recursos

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Custom Tools](https://python.langchain.com/docs/modules/tools/custom_tools)

---

## 🚨 Bloqueadores Conhecidos

1. Se `AsyncRAGEngine` não tiver método `search_documents`, você precisa adicioná-lo
2. Se `CodeExampleGenerator` não existir, crie uma versão simplificada
3. LangChain pode ter breaking changes entre versões - use versão fixa

---

## 🤝 Dependências de Outros Devs

**Nenhuma** - você pode começar imediatamente!

Outros devs dependem de VOCÊ:
- DEV_3 (Observability) precisa do agent funcionando
- DEV_4 (Streaming) precisa do agent funcionando
- DEV_5 (Evaluation) precisa do agent funcionando

**Você é o bloqueador crítico - comece o quanto antes!** ⚡

---

**Boa sorte, DEV_1! 🚀**
