# Project Standards

This document outlines the coding standards, naming conventions, and best practices for the DocRag project.

## Code Style and Formatting

### Python Standards

- **Python Version**: Python 3.11+
- **Line Length**: Maximum 88 characters (Black formatter standard)
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Double quotes for strings, single quotes for string literals in f-strings
- **Import Organization**: Standard library, third-party, local imports (separated by blank lines)

### Import Standards

```python
# Standard library imports
import os
import logging
from typing import List, Dict, Any, Optional

# Third-party imports
from flask import Flask, request, jsonify
import chromadb
from openai import OpenAI

# Local imports
from config import Config
from models import Conversation
```

### Docstring Standards

Use Google-style docstrings for all classes and functions:

```python
def validate_query(query: str, max_length: int = 500) -> Dict[str, Any]:
    """
    Validate user query input and sanitize for security.
    
    Args:
        query: User input query string
        max_length: Maximum allowed query length
        
    Returns:
        Dictionary containing validation results with keys:
        - valid: Boolean indicating if query is valid
        - query: Cleaned query string
        - errors: List of validation errors
        
    Raises:
        ValidationError: When query contains malicious content
    """
```

## Naming Conventions

### Files and Directories

- **Python files**: `snake_case.py`
- **Configuration files**: `snake_case.py`, `kebab-case.yml`
- **Directories**: `snake_case` or `kebab-case`
- **Templates**: `kebab-case.html`

### Variables and Functions

- **Variables**: `snake_case`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Class names**: `PascalCase`
- **Private methods**: `_leading_underscore`

### Database

- **Table names**: `snake_case` (singular)
- **Column names**: `snake_case`
- **Index names**: `idx_{table}_{column}` or `idx_{table}_{column1}_{column2}`
- **Foreign keys**: `{referenced_table}_id`

## Code Organization Principles

### File Size Limits

- **Python modules**: Maximum ~300 lines
- **Functions**: Maximum ~50 lines
- **Classes**: Maximum ~200 lines

### Module Structure

```python
"""Module docstring explaining purpose"""

# Imports (standard, third-party, local)
import os
from typing import Dict

from flask import Flask
from config import Config

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Classes
class ExampleClass:
    """Class implementation"""
    pass

# Functions
def example_function():
    """Function implementation"""
    pass

# Main execution (if applicable)
if __name__ == "__main__":
    pass
```

### Error Handling Standards

```python
# Use specific exceptions
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise ProcessingError(f"Failed to process: {e}") from e
except Exception as e:
    logger.exception("Unexpected error occurred")
    raise

# Always log errors with context
logger.error(f"Failed to process document {doc_id}: {str(e)}")
```

## Configuration Management

### Environment Variables

- **Naming**: `UPPER_SNAKE_CASE`
- **Grouping**: Prefix related variables (e.g., `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`)
- **Documentation**: All variables documented in `.env.example`

### Configuration Access

```python
# Centralized configuration access
from config import Config

# Good
api_key = Config.OPENAI_API_KEY
timeout = Config.OPENAI_TIMEOUT

# Bad - direct os.getenv access
api_key = os.getenv("OPENAI_API_KEY")
```

## Logging Standards

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages
- **WARNING**: Potential issues that don't break functionality
- **ERROR**: Errors that break specific operations
- **CRITICAL**: Errors that may crash the application

### Log Format

```python
import logging

logger = logging.getLogger(__name__)

# Good logging examples
logger.info(f"Processing document: {doc_url}")
logger.warning(f"Rate limit exceeded for IP {ip_address}")
logger.error(f"Failed to embed chunk {chunk_id}: {str(e)}")
```

## Database Standards

### Model Design

```python
class DocumentChunk(db.Model):
    """Represents a processed document chunk with metadata."""
    
    # Primary key
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Required fields
    source_url = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    
    # Optional fields with defaults
    chunk_index = db.Column(db.Integer, nullable=False, default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### Migration Standards

- **File naming**: `YYYY_MM_DD_HHMMSS_description.py`
- **Idempotent**: Migrations should be safe to run multiple times
- **Rollback**: Include downgrade methods when possible
- **Data migration**: Separate data migrations from schema changes

## Security Standards

### Input Validation

```python
from utils.validators import validate_query, sanitize_input

# Always validate user input
validation_result = validate_query(user_query)
if not validation_result['valid']:
    return jsonify({'error': validation_result['errors']}), 400

clean_query = validation_result['query']
```

### Secret Management

- **Never commit secrets**: Use `.env.example` with placeholders
- **Environment variables**: Store all secrets in environment variables
- **Validation**: Validate required secrets on startup

```python
# Good
@classmethod
def validate_config(cls):
    if not cls.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
```

## Performance Standards

### Database Queries

- **N+1 Prevention**: Use joins or batch queries
- **Indexing**: Index frequently queried columns
- **Connection Pooling**: Configure appropriate pool sizes

### Caching

```python
# Use caching for expensive operations
cache_key = f"embedding:{content_hash}"
cached_result = cache.get(cache_key)
if cached_result is None:
    result = expensive_operation()
    cache.set(cache_key, result, ttl=Config.CACHE_TTL)
    return result
return cached_result
```

### Resource Management

```python
# Use context managers for resources
with open(file_path, 'r') as f:
    content = f.read()

# Close database connections properly
try:
    result = db.session.execute(query)
    return result.fetchall()
finally:
    db.session.close()
```

## Testing Standards

### Test File Organization

- **Test files**: `test_*.py` or `*_test.py`
- **Test classes**: `Test{ClassName}`
- **Test methods**: `test_{functionality}_when_{condition}_should_{expected_outcome}`

### Test Examples

```python
def test_validate_query_when_valid_input_should_return_clean_query():
    """Test query validation with valid input."""
    result = validate_query("What is React?")
    
    assert result['valid'] is True
    assert result['query'] == "What is React?"
    assert len(result['errors']) == 0

def test_validate_query_when_malicious_input_should_reject():
    """Test query validation rejects malicious content."""
    malicious_query = "<script>alert('xss')</script>"
    result = validate_query(malicious_query)
    
    assert result['valid'] is False
    assert 'malicious content' in str(result['errors'])
```

## Documentation Standards

### Code Comments

```python
# Comments explain WHY, not WHAT
# Normalize legacy postgres:// URLs for SQLAlchemy compatibility
if uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)

# Use inline comments sparingly
response_time = time.time() - start_time  # Track operation duration
```

### README Standards

- **Structure**: Title, Summary, Contents, sections
- **Code examples**: Include working examples
- **Environment**: Document all environment variables
- **Deployment**: Step-by-step deployment instructions

## Quality Checklist

Before submitting code, ensure:

- [ ] Code follows naming conventions
- [ ] Functions have docstrings
- [ ] Input validation is implemented
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate for the operation
- [ ] Tests cover new functionality
- [ ] No secrets are committed
- [ ] Database migrations are included if needed
- [ ] Performance implications are considered
- [ ] Documentation is updated

## Tools and Automation

### Recommended Tools

- **Code Formatting**: Black (line length 88)
- **Import Sorting**: isort
- **Type Checking**: mypy
- **Linting**: flake8 or ruff
- **Testing**: pytest
- **Coverage**: pytest-cov

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml example
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        args: [--line-length=88]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
```

---

*This document should be reviewed and updated as the project evolves. All team members are expected to follow these standards.*