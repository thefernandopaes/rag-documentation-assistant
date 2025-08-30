# Testing Strategy

This document outlines the testing approaches, frameworks, and coverage requirements for the DocRag application.

## Testing Philosophy

Our testing strategy follows the **test pyramid** approach:
- **Unit Tests** (70%): Fast, isolated, comprehensive
- **Integration Tests** (20%): Component interactions, database operations
- **End-to-End Tests** (10%): Full user workflows

## Testing Framework

### Primary Tools

- **pytest**: Main testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **flask-testing**: Flask-specific testing utilities

### Installation

```bash
pip install pytest pytest-cov pytest-mock flask-testing
```

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_validators.py
│   ├── test_config.py
│   ├── test_cache_manager.py
│   ├── test_rag_engine.py
│   └── test_models.py
├── integration/                # Integration tests (database, API)
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   ├── test_rag_workflow.py
│   └── test_document_processing.py
├── e2e/                       # End-to-end tests
│   └── test_user_workflows.py
└── fixtures/                  # Test data and fixtures
    ├── sample_documents.json
    ├── sample_responses.json
    └── test_data.py
```

### Test Configuration

**conftest.py**:
```python
import pytest
import tempfile
import os
from app import create_app, db
from config import Config

@pytest.fixture(scope='session')
def app():
    """Create application for testing."""
    # Use test configuration
    test_config = {
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'CHROMA_DB_PATH': tempfile.mkdtemp(),
        'OPENAI_API_KEY': 'test-key',
        'SESSION_SECRET': 'test-secret',
        'DOC_USE_SAMPLE': True,
        'CACHE_TTL': 1,  # Short cache for testing
    }
    
    # Override config for testing
    for key, value in test_config.items():
        setattr(Config, key, value)
    
    app = create_app()
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def db_session(app):
    """Create database session for testing."""
    with app.app_context():
        yield db.session
        db.session.rollback()

@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing."""
    return {
        'session_id': 'test-session-123',
        'user_query': 'What is React?',
        'ai_response': 'React is a JavaScript library for building user interfaces.',
        'sources': '["https://react.dev/learn"]',
        'response_time': 1.234
    }
```

## Unit Testing

### Testing Validation Logic

```python
# tests/unit/test_validators.py
import pytest
from utils.validators import validate_query, sanitize_input, ValidationError

class TestQueryValidation:
    """Test query validation functionality."""
    
    def test_validate_query_with_valid_input_returns_success(self):
        """Test validation succeeds with valid input."""
        result = validate_query("What is Python?")
        
        assert result['valid'] is True
        assert result['query'] == "What is Python?"
        assert len(result['errors']) == 0
    
    def test_validate_query_with_empty_input_returns_error(self):
        """Test validation fails with empty input."""
        result = validate_query("")
        
        assert result['valid'] is False
        assert 'cannot be empty' in str(result['errors'])
    
    def test_validate_query_with_xss_input_returns_error(self):
        """Test validation rejects XSS attempts."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for malicious_input in malicious_inputs:
            result = validate_query(malicious_input)
            assert result['valid'] is False
            assert 'malicious content' in str(result['errors'])
    
    def test_validate_query_with_sql_injection_returns_warning(self):
        """Test validation detects SQL injection attempts."""
        sql_inputs = [
            "'; DROP TABLE users; --",
            "SELECT * FROM users",
            "UNION SELECT password FROM users"
        ]
        
        for sql_input in sql_inputs:
            result = validate_query(sql_input)
            # Should either be rejected or flagged with warning
            assert result['valid'] is False or len(result['warnings']) > 0

class TestInputSanitization:
    """Test input sanitization functionality."""
    
    def test_sanitize_input_removes_html_tags(self):
        """Test HTML tag removal."""
        dirty_input = "<b>Hello</b> <script>alert('test')</script>World"
        clean_output = sanitize_input(dirty_input)
        
        assert '<' not in clean_output
        assert '>' not in clean_output
        assert 'Hello' in clean_output
        assert 'World' in clean_output
    
    def test_sanitize_input_normalizes_whitespace(self):
        """Test whitespace normalization."""
        dirty_input = "Hello    \n\n   World   \t\t"
        clean_output = sanitize_input(dirty_input)
        
        assert clean_output == "Hello World"
    
    def test_sanitize_input_preserves_newlines_when_requested(self):
        """Test newline preservation option."""
        input_with_newlines = "Line 1\nLine 2\nLine 3"
        clean_output = sanitize_input(input_with_newlines, preserve_newlines=True)
        
        assert "\n" in clean_output
        assert clean_output.count('\n') == 2
```

### Testing Configuration

```python
# tests/unit/test_config.py
import pytest
import os
from config import Config

class TestConfig:
    """Test configuration management."""
    
    def test_config_validation_with_missing_api_key_raises_error(self):
        """Test configuration validation fails without API key."""
        original_key = Config.OPENAI_API_KEY
        Config.OPENAI_API_KEY = None
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY.*required"):
            Config.validate_config()
        
        Config.OPENAI_API_KEY = original_key
    
    def test_database_uri_normalization_converts_postgres_scheme(self):
        """Test database URI normalization."""
        original_uri = Config.DATABASE_URL
        Config.DATABASE_URL = "postgres://user:pass@host:5432/db"
        
        normalized = Config.get_database_uri()
        assert normalized.startswith("postgresql+psycopg2://")
        
        Config.DATABASE_URL = original_uri
    
    def test_production_detection_identifies_production_environment(self):
        """Test production environment detection."""
        original_env = os.getenv('ENV')
        
        os.environ['ENV'] = 'production'
        assert Config._is_production() is True
        
        os.environ['ENV'] = 'development'
        assert Config._is_production() is False
        
        # Restore original
        if original_env:
            os.environ['ENV'] = original_env
        elif 'ENV' in os.environ:
            del os.environ['ENV']
```

### Testing Models

```python
# tests/unit/test_models.py
import pytest
from datetime import datetime
from models import Conversation, DocumentChunk, RateLimit

class TestConversationModel:
    """Test Conversation model functionality."""
    
    def test_conversation_creation_with_valid_data_succeeds(self, db_session):
        """Test creating conversation with valid data."""
        conversation = Conversation(
            session_id='test-session',
            user_query='Test query',
            ai_response='Test response'
        )
        
        db_session.add(conversation)
        db_session.commit()
        
        assert conversation.id is not None
        assert conversation.created_at is not None
        assert isinstance(conversation.created_at, datetime)
    
    def test_conversation_query_by_session_returns_correct_results(self, db_session):
        """Test querying conversations by session ID."""
        session_id = 'test-session-123'
        
        # Create test conversations
        conv1 = Conversation(session_id=session_id, user_query='Query 1', ai_response='Response 1')
        conv2 = Conversation(session_id=session_id, user_query='Query 2', ai_response='Response 2')
        conv3 = Conversation(session_id='other-session', user_query='Query 3', ai_response='Response 3')
        
        db_session.add_all([conv1, conv2, conv3])
        db_session.commit()
        
        # Query by session
        results = Conversation.query.filter_by(session_id=session_id).all()
        
        assert len(results) == 2
        assert all(conv.session_id == session_id for conv in results)

class TestDocumentChunkModel:
    """Test DocumentChunk model functionality."""
    
    def test_document_chunk_with_unique_content_hash_succeeds(self, db_session):
        """Test creating document chunk with unique content hash."""
        chunk = DocumentChunk(
            source_url='https://example.com/doc',
            content='Test content',
            chunk_index=0,
            content_hash='unique-hash-123'
        )
        
        db_session.add(chunk)
        db_session.commit()
        
        assert chunk.id is not None
    
    def test_document_chunk_with_duplicate_hash_fails(self, db_session):
        """Test duplicate content hash constraint."""
        hash_value = 'duplicate-hash'
        
        # Create first chunk
        chunk1 = DocumentChunk(
            source_url='https://example.com/doc1',
            content='Content 1',
            chunk_index=0,
            content_hash=hash_value
        )
        db_session.add(chunk1)
        db_session.commit()
        
        # Attempt to create duplicate
        chunk2 = DocumentChunk(
            source_url='https://example.com/doc2',
            content='Content 2',
            chunk_index=0,
            content_hash=hash_value  # Same hash
        )
        db_session.add(chunk2)
        
        # Should raise integrity error
        with pytest.raises(Exception):  # IntegrityError or similar
            db_session.commit()
```

## Integration Testing

### API Endpoint Testing

```python
# tests/integration/test_api_endpoints.py
import json
import pytest
from models import Conversation

class TestChatAPI:
    """Test chat API endpoint integration."""
    
    def test_chat_api_with_valid_query_returns_response(self, client):
        """Test successful chat API request."""
        response = client.post('/api/chat',
                              json={'query': 'What is React?'},
                              content_type='application/json')
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'response' in data
        assert 'sources' in data
        assert 'session_id' in data
        assert 'response_time' in data
        
        # Verify response structure
        assert isinstance(data['response'], str)
        assert isinstance(data['sources'], list)
        assert len(data['response']) > 0
    
    def test_chat_api_with_invalid_query_returns_error(self, client):
        """Test chat API with invalid input."""
        response = client.post('/api/chat',
                              json={'query': ''},  # Empty query
                              content_type='application/json')
        
        assert response.status_code == 400
        
        data = response.get_json()
        assert 'error' in data
        assert 'code' in data
    
    def test_chat_api_stores_conversation_in_database(self, client, db_session):
        """Test that chat API stores conversation in database."""
        initial_count = Conversation.query.count()
        
        response = client.post('/api/chat',
                              json={'query': 'Test query'},
                              content_type='application/json')
        
        assert response.status_code == 200
        
        # Verify conversation was stored
        final_count = Conversation.query.count()
        assert final_count == initial_count + 1
        
        # Verify conversation content
        conversation = Conversation.query.order_by(Conversation.created_at.desc()).first()
        assert conversation.user_query == 'Test query'
        assert len(conversation.ai_response) > 0

class TestStatsAPI:
    """Test statistics API endpoint."""
    
    def test_stats_api_returns_system_statistics(self, client):
        """Test stats API returns valid statistics."""
        response = client.get('/api/stats')
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'stats' in data
        assert 'total_documents' in data['stats']
        assert 'total_conversations' in data['stats']
        assert 'cache_stats' in data['stats']

class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_healthz_endpoint_returns_healthy_status(self, client):
        """Test health check returns healthy status."""
        response = client.get('/healthz')
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'checks' in data
        assert 'database' in data['checks']
```

### Database Integration Testing

```python
# tests/integration/test_database_operations.py
import pytest
from datetime import datetime, timedelta
from models import Conversation, DocumentChunk, RateLimit

class TestDatabaseOperations:
    """Test database operations and constraints."""
    
    def test_conversation_cascade_operations(self, db_session):
        """Test conversation database operations."""
        session_id = 'test-session'
        
        # Create multiple conversations
        conversations = [
            Conversation(session_id=session_id, user_query=f'Query {i}', 
                        ai_response=f'Response {i}')
            for i in range(3)
        ]
        
        db_session.add_all(conversations)
        db_session.commit()
        
        # Test bulk operations
        results = Conversation.query.filter_by(session_id=session_id).all()
        assert len(results) == 3
        
        # Test deletion
        Conversation.query.filter_by(session_id=session_id).delete()
        db_session.commit()
        
        remaining = Conversation.query.filter_by(session_id=session_id).count()
        assert remaining == 0
    
    def test_document_chunk_idempotency(self, db_session):
        """Test document chunk idempotency via content hash."""
        content_hash = 'test-hash-123'
        
        # Create first chunk
        chunk1 = DocumentChunk(
            source_url='https://example.com',
            content='Test content',
            chunk_index=0,
            content_hash=content_hash
        )
        db_session.add(chunk1)
        db_session.commit()
        
        # Attempt to create duplicate
        chunk2 = DocumentChunk(
            source_url='https://example.com/other',
            content='Different content',
            chunk_index=1,
            content_hash=content_hash  # Same hash
        )
        db_session.add(chunk2)
        
        # Should fail due to unique constraint
        with pytest.raises(Exception):
            db_session.commit()
    
    def test_rate_limit_window_reset(self, db_session):
        """Test rate limit window reset functionality."""
        ip_address = '192.168.1.100'
        
        # Create rate limit record
        rate_limit = RateLimit(
            ip_address=ip_address,
            request_count=5,
            reset_time=datetime.utcnow() - timedelta(minutes=2)  # Expired
        )
        db_session.add(rate_limit)
        db_session.commit()
        
        # Simulate rate limit check
        existing = RateLimit.query.filter_by(ip_address=ip_address).first()
        assert existing is not None
        
        # Check if reset needed
        if datetime.utcnow() >= existing.reset_time:
            existing.request_count = 0
            existing.reset_time = datetime.utcnow() + timedelta(minutes=1)
            db_session.commit()
        
        # Verify reset
        updated = RateLimit.query.filter_by(ip_address=ip_address).first()
        assert updated.request_count == 0
        assert updated.reset_time > datetime.utcnow()
```

### RAG Engine Integration Testing

```python
# tests/integration/test_rag_workflow.py
import pytest
from unittest.mock import Mock, patch
from rag_engine import RAGEngine

class TestRAGWorkflow:
    """Test RAG engine integration workflows."""
    
    @pytest.fixture
    def rag_engine(self):
        """Create RAG engine for testing."""
        with patch('rag_engine.OpenAI') as mock_openai:
            # Mock OpenAI client
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock embedding response
            mock_embedding_response = Mock()
            mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_embedding_response
            
            # Mock chat completion response
            mock_chat_response = Mock()
            mock_chat_response.choices = [
                Mock(message=Mock(content='{"answer": "Test response", "sources": [], "examples": []}'))
            ]
            mock_client.chat.completions.create.return_value = mock_chat_response
            
            with patch('rag_engine.chromadb.PersistentClient') as mock_chroma:
                # Mock ChromaDB
                mock_collection = Mock()
                mock_collection.query.return_value = {
                    'documents': [['Test document content']],
                    'metadatas': [[{'source_url': 'https://example.com', 'title': 'Test Doc'}]],
                    'distances': [[0.2]]
                }
                
                mock_client_instance = Mock()
                mock_client_instance.get_or_create_collection.return_value = mock_collection
                mock_chroma.return_value = mock_client_instance
                
                engine = RAGEngine()
                engine.collection = mock_collection
                yield engine
    
    def test_query_processing_end_to_end(self, rag_engine):
        """Test complete query processing workflow."""
        query = "What is React?"
        
        # Process query
        result = rag_engine.query(query)
        
        # Verify response structure
        assert 'answer' in result
        assert 'sources' in result
        assert 'response_time' in result
        
        # Verify OpenAI was called
        rag_engine.openai_client.embeddings.create.assert_called()
        rag_engine.openai_client.chat.completions.create.assert_called()
        
        # Verify ChromaDB was queried
        rag_engine.collection.query.assert_called()
    
    def test_document_addition_workflow(self, rag_engine):
        """Test document addition and embedding workflow."""
        documents = [{
            'content': 'Test document content',
            'source_url': 'https://example.com/test',
            'title': 'Test Document',
            'chunk_index': 0,
            'doc_type': 'test'
        }]
        
        # Add documents
        rag_engine.add_documents(documents)
        
        # Verify embedding generation was called
        rag_engine.openai_client.embeddings.create.assert_called()
        
        # Verify ChromaDB addition was called
        rag_engine.collection.add.assert_called()
    
    def test_caching_behavior(self, rag_engine):
        """Test query result caching."""
        query = "What is caching?"
        
        # First query (cache miss)
        result1 = rag_engine.query(query)
        assert result1['cached'] is False
        
        # Second query (cache hit)
        result2 = rag_engine.query(query)
        assert result2['cached'] is True
        
        # Results should be identical
        assert result1['answer'] == result2['answer']
```

## End-to-End Testing

### User Workflow Testing

```python
# tests/e2e/test_user_workflows.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestUserWorkflows:
    """Test complete user workflows using browser automation."""
    
    @pytest.fixture(scope='class')
    def driver(self):
        """Create browser driver for testing."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()
    
    def test_complete_chat_workflow(self, driver, app):
        """Test complete chat interaction workflow."""
        # Navigate to chat page
        driver.get('http://localhost:5000/chat')
        
        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        chat_input = wait.until(EC.presence_of_element_located((By.ID, 'query-input')))
        
        # Enter query
        chat_input.send_keys('What is React?')
        
        # Submit query
        submit_button = driver.find_element(By.ID, 'submit-button')
        submit_button.click()
        
        # Wait for response
        response_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, 'ai-response'))
        )
        
        # Verify response contains content
        response_text = response_element.text
        assert len(response_text) > 0
        assert 'React' in response_text
        
        # Verify sources are displayed
        sources = driver.find_elements(By.CLASS_NAME, 'source-link')
        assert len(sources) > 0
    
    def test_rate_limiting_behavior(self, driver, app):
        """Test rate limiting prevents abuse."""
        driver.get('http://localhost:5000/chat')
        
        # Rapidly submit multiple queries
        for i in range(15):  # Exceed rate limit
            query_input = driver.find_element(By.ID, 'query-input')
            query_input.clear()
            query_input.send_keys(f'Query {i}')
            
            submit_button = driver.find_element(By.ID, 'submit-button')
            submit_button.click()
            
            # Short delay
            time.sleep(0.1)
        
        # Check for rate limit message
        wait = WebDriverWait(driver, 5)
        try:
            rate_limit_message = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, 'rate-limit-error'))
            )
            assert 'rate limit' in rate_limit_message.text.lower()
        except:
            pytest.fail("Rate limiting not triggered as expected")
```

## Test Data Management

### Fixtures and Test Data

```python
# tests/fixtures/test_data.py
import json
from typing import List, Dict

def get_sample_documents() -> List[Dict]:
    """Get sample documents for testing."""
    return [
        {
            'source_url': 'https://react.dev/learn/describing-the-ui',
            'title': 'Describing the UI',
            'content': 'React is a JavaScript library for building user interfaces...',
            'doc_type': 'react',
            'chunk_index': 0
        },
        {
            'source_url': 'https://docs.python.org/3/tutorial/introduction.html',
            'title': 'An Informal Introduction to Python',
            'content': 'Python is an easy to learn, powerful programming language...',
            'doc_type': 'python',
            'chunk_index': 0
        }
    ]

def get_sample_conversations() -> List[Dict]:
    """Get sample conversations for testing."""
    return [
        {
            'session_id': 'test-session-1',
            'user_query': 'What is React?',
            'ai_response': 'React is a JavaScript library for building user interfaces.',
            'sources': '["https://react.dev/learn"]'
        },
        {
            'session_id': 'test-session-1',
            'user_query': 'How do I create components?',
            'ai_response': 'You can create React components using function or class syntax.',
            'sources': '["https://react.dev/learn/your-first-component"]'
        }
    ]
```

### Database Test Utilities

```python
# tests/fixtures/db_utils.py
from models import Conversation, DocumentChunk
from app import db

def create_test_conversations(session_id: str, count: int = 3):
    """Create test conversations for testing."""
    conversations = []
    
    for i in range(count):
        conv = Conversation(
            session_id=session_id,
            user_query=f'Test query {i+1}',
            ai_response=f'Test response {i+1}',
            response_time=1.0 + i * 0.1
        )
        conversations.append(conv)
    
    db.session.add_all(conversations)
    db.session.commit()
    
    return conversations

def create_test_document_chunks(source_url: str, count: int = 5):
    """Create test document chunks."""
    chunks = []
    
    for i in range(count):
        chunk = DocumentChunk(
            source_url=source_url,
            title=f'Test Document {i+1}',
            content=f'Test content for chunk {i+1}',
            chunk_index=i,
            doc_type='test',
            content_hash=f'test-hash-{i}'
        )
        chunks.append(chunk)
    
    db.session.add_all(chunks)
    db.session.commit()
    
    return chunks
```

## Testing Configuration

### pytest Configuration

**pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test* *Tests
python_functions = test_*
addopts = 
    --verbose
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=85
    --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests (may be skipped in CI)
    external: Tests requiring external services
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Skip slow tests

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_validators.py

# Run specific test method
pytest tests/unit/test_validators.py::TestQueryValidation::test_validate_query_with_valid_input
```

### Coverage Requirements

**Minimum Coverage**: 85% overall
**Target Coverage by Component**:
- **Validators**: 95% (critical for security)
- **Models**: 90% (data integrity)
- **RAG Engine**: 85% (core functionality)
- **API Routes**: 80% (integration focus)
- **Configuration**: 90% (startup critical)

**Coverage Exclusions**:
```ini
# .coveragerc
[run]
omit = 
    .venv/*
    tests/*
    migrations/*
    scripts/*
    gunicorn.conf.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

## Performance Testing

### Load Testing

```python
# tests/performance/test_load.py
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def make_request(session, query):
    """Make async request to chat API."""
    async with session.post('http://localhost:5000/api/chat',
                           json={'query': query}) as response:
        return await response.json()

async def load_test_chat_api(concurrent_users=10, requests_per_user=5):
    """Load test the chat API endpoint."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for user in range(concurrent_users):
            for request in range(requests_per_user):
                query = f"Load test query {user}-{request}"
                task = make_request(session, query)
                tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        
        print(f"Load test results:")
        print(f"Total requests: {len(tasks)}")
        print(f"Successful: {successful_requests}")
        print(f"Failed: {len(tasks) - successful_requests}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests per second: {len(tasks) / total_time:.2f}")

if __name__ == "__main__":
    asyncio.run(load_test_chat_api())
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_docrag
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov pytest-mock
        pip install -r <(python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=. --cov-report=xml
      env:
        OPENAI_API_KEY: test-key
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_docrag
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        OPENAI_API_KEY: test-key
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_docrag
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Testing Best Practices

### Test Writing Guidelines

1. **Test Naming**: Use descriptive names that explain the scenario
   ```python
   def test_rate_limiter_when_limit_exceeded_should_reject_request():
   ```

2. **Test Structure**: Follow Arrange-Act-Assert pattern
   ```python
   def test_example():
       # Arrange
       setup_data = create_test_data()
       
       # Act
       result = function_under_test(setup_data)
       
       # Assert
       assert result.status == 'success'
   ```

3. **Isolation**: Each test should be independent
   ```python
   def test_isolated_behavior(db_session):
       # Use fresh database session
       # Clean up after test automatically
   ```

4. **Mocking**: Mock external dependencies
   ```python
   @patch('rag_engine.OpenAI')
   def test_without_external_api(mock_openai):
       # Test logic without hitting real API
   ```

### Common Pitfalls

**Avoid**:
- Tests that depend on external services
- Tests that modify global state
- Tests with hardcoded timing assumptions
- Tests that don't clean up after themselves
- Tests that are too broad or too narrow

**Good Practices**:
- Use fixtures for common setup
- Mock external dependencies
- Test edge cases and error conditions
- Keep tests fast and focused
- Write tests for both happy path and error cases

---

*A comprehensive testing strategy ensures code quality, prevents regressions, and enables confident deployments. Invest time in good tests - they pay dividends.*