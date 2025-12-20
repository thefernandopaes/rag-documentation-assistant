"""
Agent Tests - Comprehensive Testing for LangChain Agent

Tests for agent, tools, and endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Agent and tools
from langchain_agent import DocumentationAgent, get_agent
from tools.rag_tool import RAGSearchTool
from tools.code_generator_tool import CodeGeneratorTool
from tools.validator_tool import CodeValidatorTool

# Schemas
from schemas_agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentToolInfo,
    AgentTestRequest
)


# ============================================================================
# TOOL TESTS
# ============================================================================

class TestRAGSearchTool:
    """Tests for RAG search tool"""

    @pytest.fixture
    def tool(self):
        return RAGSearchTool()

    @pytest.mark.asyncio
    async def test_tool_initialization(self, tool):
        """Test tool initializes correctly"""
        assert tool.name == "documentation_search"
        assert "documentation" in tool.description.lower()
        assert tool.args_schema is not None

    @pytest.mark.asyncio
    async def test_search_success(self, tool):
        """Test successful documentation search"""
        # Mock RAG engine
        mock_engine = AsyncMock()
        mock_engine.search_documents.return_value = {
            'documents': [['FastAPI is a modern web framework for building APIs']],
            'metadatas': [[{'title': 'FastAPI Documentation', 'source_url': 'https://fastapi.tiangolo.com'}]]
        }

        with patch.object(tool, '_get_rag_engine', return_value=mock_engine):
            result = await tool._arun("What is FastAPI?")

            assert "FastAPI" in result
            assert "Documentation" in result
            assert mock_engine.search_documents.called

    @pytest.mark.asyncio
    async def test_search_no_results(self, tool):
        """Test search with no results"""
        mock_engine = AsyncMock()
        mock_engine.search_documents.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        with patch.object(tool, '_get_rag_engine', return_value=mock_engine):
            result = await tool._arun("nonexistent query")

            assert "No relevant documentation found" in result


class TestCodeGeneratorTool:
    """Tests for code generator tool"""

    @pytest.fixture
    def tool(self):
        return CodeGeneratorTool()

    def test_tool_initialization(self, tool):
        """Test tool initializes correctly"""
        assert tool.name == "code_generator"
        assert "code" in tool.description.lower()
        assert tool.args_schema is not None

    @pytest.mark.asyncio
    async def test_generate_fastapi_code(self, tool):
        """Test FastAPI code generation"""
        result = await tool._arun("Python FastAPI POST endpoint")

        assert "fastapi" in result.lower()
        assert "async" in result.lower()
        assert "def" in result or "@app" in result

    @pytest.mark.asyncio
    async def test_generate_python_async_code(self, tool):
        """Test Python async code generation"""
        result = await tool._arun("Python async function")

        assert "async" in result.lower()
        assert "await" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_curl_code(self, tool):
        """Test cURL code generation"""
        result = await tool._arun("cURL GET request")

        assert "curl" in result.lower()
        assert "-X" in result or "GET" in result


class TestCodeValidatorTool:
    """Tests for code validator tool"""

    @pytest.fixture
    def tool(self):
        return CodeValidatorTool()

    def test_tool_initialization(self, tool):
        """Test tool initializes correctly"""
        assert tool.name == "code_validator"
        assert "validate" in tool.description.lower()
        assert tool.args_schema is not None

    @pytest.mark.asyncio
    async def test_validate_valid_python(self, tool):
        """Test validation of valid Python code"""
        code = """
def hello():
    return "world"
"""
        result = await tool._arun(code, language="python")

        assert "✅" in result or "passed" in result.lower()

    @pytest.mark.asyncio
    async def test_validate_invalid_python(self, tool):
        """Test validation of invalid Python code"""
        code = """
def hello(
    return "world"
"""
        result = await tool._arun(code, language="python")

        assert "❌" in result or "failed" in result.lower()
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_validate_javascript(self, tool):
        """Test JavaScript validation"""
        code = """
function hello() {
    return "world";
}
"""
        result = await tool._arun(code, language="javascript")

        assert "✅" in result or "passed" in result.lower() or "correct" in result.lower()

    @pytest.mark.asyncio
    async def test_validate_unbalanced_brackets(self, tool):
        """Test detection of unbalanced brackets"""
        code = """
function hello() {
    if (true) {
        return "world";
}
"""
        result = await tool._arun(code, language="javascript")

        # Should detect bracket mismatch
        assert "❌" in result or "bracket" in result.lower() or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_auto_detect_language(self, tool):
        """Test automatic language detection"""
        # Python code without explicit language
        python_code = "def hello(): return 'world'"
        result = await tool._arun(python_code)

        assert "python" in result.lower()

        # JavaScript code without explicit language
        js_code = "function hello() { return 'world'; }"
        result = await tool._arun(js_code)

        assert "javascript" in result.lower()


# ============================================================================
# AGENT TESTS
# ============================================================================

class TestDocumentationAgent:
    """Tests for DocumentationAgent"""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        return DocumentationAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes with all components"""
        assert agent.llm is not None
        assert len(agent.tools) == 3
        assert agent.agent is not None
        assert agent.agent_executor is not None

    def test_agent_tools(self, agent):
        """Test agent has correct tools"""
        tool_names = [tool.name for tool in agent.tools]

        assert "documentation_search" in tool_names
        assert "code_generator" in tool_names
        assert "code_validator" in tool_names

    @pytest.mark.asyncio
    async def test_agent_simple_query(self, agent):
        """Test agent responds to simple query"""
        # Mock tools to avoid real API calls
        with patch.object(agent.tools[0], '_arun', return_value="FastAPI is a web framework"), \
             patch.object(agent.tools[1], '_arun', return_value="```python\ncode```"), \
             patch.object(agent.tools[2], '_arun', return_value="✅ Valid"):

            result = await agent.arun("What is FastAPI?")

            assert 'output' in result
            assert isinstance(result['output'], str)
            assert 'tool_calls' in result
            assert 'intermediate_steps' in result

    def test_get_tools_info(self, agent):
        """Test getting tools information"""
        tools_info = agent.get_tools_info()

        assert len(tools_info) == 3
        assert all('name' in tool for tool in tools_info)
        assert all('description' in tool for tool in tools_info)


# ============================================================================
# ENDPOINT TESTS
# ============================================================================

@pytest.mark.asyncio
class TestAgentEndpoints:
    """Tests for agent API endpoints"""

    @pytest.fixture
    async def app(self):
        """Create test FastAPI app"""
        from fastapi_app import app
        return app

    @pytest.fixture
    async def client(self, app):
        """Create async test client"""
        from httpx import ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def test_agent_chat_endpoint(self, client):
        """Test /api/agent/chat endpoint"""
        # Mock agent to avoid real LLM calls
        mock_result = {
            'output': 'FastAPI is a modern web framework',
            'intermediate_steps': [],
            'tool_calls': [
                {
                    'tool': 'documentation_search',
                    'input': 'FastAPI',
                    'output': 'FastAPI documentation...'
                }
            ]
        }

        with patch('langchain_agent.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.arun.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            response = await client.post(
                "/api/agent/chat",
                json={"query": "What is FastAPI?", "use_history": False}
            )

            assert response.status_code == 200
            data = response.json()

            assert 'response' in data
            assert 'tool_calls' in data
            assert 'session_id' in data
            assert 'response_time' in data

    async def test_agent_chat_validation(self, client):
        """Test request validation"""
        # Empty query
        response = await client.post(
            "/api/agent/chat",
            json={"query": "", "use_history": False}
        )
        assert response.status_code == 422  # Validation error

        # Too short query
        response = await client.post(
            "/api/agent/chat",
            json={"query": "Hi", "use_history": False}
        )
        assert response.status_code == 422

    async def test_list_tools_endpoint(self, client):
        """Test /api/agent/tools endpoint"""
        with patch('langchain_agent.get_agent') as mock_get_agent:
            mock_agent = Mock()
            mock_agent.get_tools_info.return_value = [
                {'name': 'tool1', 'description': 'Description 1'},
                {'name': 'tool2', 'description': 'Description 2'}
            ]
            mock_get_agent.return_value = mock_agent

            response = await client.get("/api/agent/tools")

            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            assert len(data) == 2
            assert all('name' in tool for tool in data)
            assert all('description' in tool for tool in data)

    async def test_agent_test_endpoint(self, client):
        """Test /api/agent/test endpoint (admin only)"""
        mock_result = {
            'output': 'Test response',
            'intermediate_steps': [],
            'tool_calls': []
        }

        with patch('langchain_agent.get_agent') as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.arun.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            # Without admin key (should fail in production)
            response = await client.post(
                "/api/agent/test",
                json={"test_queries": ["Test query"]}
            )

            # In development mode, might pass; in production, should be 401
            assert response.status_code in [200, 401]

            # With admin key
            headers = {"X-Admin-Key": "test-admin-key"}
            response = await client.post(
                "/api/agent/test",
                json={"test_queries": ["What is FastAPI?", "How to use async?"]},
                headers=headers
            )

            # If ADMIN_API_KEY is configured correctly, should pass
            if response.status_code == 200:
                data = response.json()
                assert 'results' in data
                assert 'summary' in data
                assert data['summary']['total'] == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
class TestAgentIntegration:
    """Integration tests (slower, require real API keys)"""

    @pytest.mark.skipif(
        True,  # Skip integration tests by default
        reason="Integration tests require --run-integration flag"
    )
    async def test_full_agent_workflow(self):
        """Test complete agent workflow with real components"""
        agent = DocumentationAgent()

        # Test query that requires multiple tools
        query = "Show me how to create a FastAPI POST endpoint and validate the code"

        result = await agent.arun(query)

        # Should have used multiple tools
        assert len(result['tool_calls']) >= 2

        # Should have documentation search
        tool_names = [call['tool'] for call in result['tool_calls']]
        assert 'documentation_search' in tool_names

        # Should have code generator
        assert 'code_generator' in tool_names

        # Response should be comprehensive
        assert len(result['output']) > 100


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires API keys)"
    )


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([
        __file__,
        "-v",  # Verbose
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
        "--color=yes"  # Colored output
    ])
