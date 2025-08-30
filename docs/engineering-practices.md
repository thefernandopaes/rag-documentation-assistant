# Engineering Practices

This document outlines the development workflow, pull request guidelines, and quality assurance processes for the DocRag project.

## Development Workflow

### Git Workflow

We follow a **simplified Git workflow** based on trunk-based development:

```
main branch (production-ready)
├── feature/add-new-endpoint
├── fix/memory-leak-in-embeddings
├── docs/update-api-documentation
└── hotfix/security-patch
```

#### Branch Naming Convention

- **Feature branches**: `feature/description-with-hyphens`
- **Bug fixes**: `fix/description-with-hyphens`
- **Documentation**: `docs/description-with-hyphens`
- **Hotfixes**: `hotfix/description-with-hyphens`
- **Refactoring**: `refactor/description-with-hyphens`

#### Commit Message Format

Follow the **Conventional Commits** specification:

```
type(scope): description

[optional body]

[optional footer(s)]
```

**Examples**:
```
feat(api): add conversation history endpoint

Add GET /api/conversations/{session_id} to retrieve chat history
for a specific session with pagination support.

Closes #123
```

```
fix(rag): prevent memory leak in embedding cache

Clear embedding cache after 1000 entries to prevent unbounded
memory growth during document processing.

Fixes #456
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without functionality changes
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks, dependency updates

### Development Process

#### 1. Issue Creation

Before starting work, create an issue that describes:

- **Problem**: What needs to be solved?
- **Solution**: Proposed approach (if known)
- **Acceptance Criteria**: How to verify completion
- **Priority**: Critical, High, Medium, Low

**Issue Template**:
```markdown
## Problem
Brief description of the issue or feature request.

## Proposed Solution
How should this be addressed?

## Acceptance Criteria
- [ ] Criteria 1
- [ ] Criteria 2
- [ ] Tests added
- [ ] Documentation updated

## Priority
- [ ] Critical (blocks release)
- [ ] High (important for next release)
- [ ] Medium (nice to have)
- [ ] Low (future consideration)
```

#### 2. Branch Creation

```bash
# Create and switch to feature branch
git checkout -b feature/user-authentication

# Link to issue in first commit
git commit -m "feat(auth): initialize user authentication system

Refs #123"
```

#### 3. Development Guidelines

**Small, Focused Changes**:
- Keep PRs small (<400 lines when possible)
- One feature/fix per PR
- Include tests with functionality changes

**Code Quality Standards**:
- Follow [Project Standards](./project-standards.md)
- Write self-documenting code
- Add comments for complex business logic
- Include docstrings for public functions

**Testing Requirements**:
- Unit tests for new functions
- Integration tests for API endpoints
- Update existing tests when changing behavior

#### 4. Pre-commit Checklist

Before committing changes:

- [ ] Code follows project standards
- [ ] Tests pass locally
- [ ] No sensitive information in commits
- [ ] Commit message follows convention
- [ ] Documentation updated if needed

### Code Review Process

#### Pull Request Creation

**PR Title**: Use conventional commit format
```
feat(api): add rate limiting to chat endpoint
```

**PR Description Template**:
```markdown
## Summary
Brief description of changes and why they're needed.

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Documentation
- [ ] Code comments added
- [ ] API documentation updated
- [ ] README updated (if needed)

## Checklist
- [ ] Code follows project standards
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered
- [ ] Error handling implemented
- [ ] Logging added where appropriate

Closes #[issue-number]
```

#### Review Requirements

**All PRs must have**:
- At least one code review approval
- All CI checks passing
- No merge conflicts
- Updated tests (if applicable)

**Review Focus Areas**:

1. **Functionality**
   - Does the code do what it's supposed to do?
   - Are edge cases handled?
   - Is error handling comprehensive?

2. **Security**
   - No hardcoded secrets or credentials
   - Input validation implemented
   - SQL injection prevention
   - XSS prevention measures

3. **Performance**
   - Efficient algorithms used
   - Database queries optimized
   - Caching implemented where beneficial
   - Memory usage reasonable

4. **Maintainability**
   - Code is readable and well-structured
   - Naming conventions followed
   - Appropriate abstraction levels
   - Documentation adequate

#### Review Process

**As a Reviewer**:

1. **Understand the Context**
   - Read the linked issue
   - Understand the problem being solved
   - Review the proposed solution approach

2. **Code Review**
   - Check functionality and logic
   - Verify security practices
   - Assess performance implications
   - Ensure code quality standards

3. **Testing Review**
   - Verify test coverage
   - Check test quality
   - Ensure tests cover edge cases

4. **Documentation Review**
   - Check if documentation is updated
   - Verify code comments are helpful
   - Ensure API changes are documented

**Providing Feedback**:

```markdown
## Functionality ✅
The implementation correctly handles user authentication.

## Security ⚠️
Consider adding rate limiting to the login endpoint to prevent brute force attacks.

## Performance ✅
Database queries are efficient and properly indexed.

## Suggestions
- Line 42: Consider extracting this validation logic into a separate function
- Consider adding integration tests for the authentication flow

## Approval
LGTM after addressing the security concern above.
```

**As a PR Author**:

1. **Respond to Feedback**
   - Address all review comments
   - Explain decisions when needed
   - Ask questions if feedback is unclear

2. **Make Changes**
   - Implement requested changes
   - Add additional tests if needed
   - Update documentation

3. **Request Re-review**
   - Mark conversations as resolved
   - Request fresh review after changes

### Continuous Integration

#### CI Pipeline

Our CI pipeline runs on every PR and includes:

1. **Code Quality Checks**
   ```yaml
   - name: Lint with flake8
     run: flake8 . --max-line-length=88
   
   - name: Format check with black
     run: black --check .
   
   - name: Type check with mypy
     run: mypy .
   ```

2. **Security Scanning**
   ```yaml
   - name: Security scan with bandit
     run: bandit -r . -f json
   
   - name: Dependency vulnerability check
     run: pip-audit
   ```

3. **Testing**
   ```yaml
   - name: Unit tests
     run: pytest tests/ -v --cov=.
   
   - name: Integration tests
     run: pytest tests/integration/ -v
   ```

4. **Database Migration Tests**
   ```yaml
   - name: Test migrations
     run: |
       alembic upgrade head
       alembic downgrade -1
       alembic upgrade head
   ```

#### Pre-commit Hooks

Setup pre-commit hooks for consistency:

```yaml
# .pre-commit-config.yaml
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

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]

  - repo: local
    hooks:
      - id: no-secrets
        name: No secrets
        entry: ./scripts/check-secrets.sh
        language: script
```

### Release Process

#### Version Numbering

We use **Semantic Versioning** (SemVer):

- **Major** (X.0.0): Breaking changes
- **Minor** (X.Y.0): New features (backward compatible)
- **Patch** (X.Y.Z): Bug fixes (backward compatible)

#### Release Workflow

1. **Prepare Release**
   ```bash
   git checkout main
   git pull origin main
   
   # Update version
   echo "1.2.3" > VERSION
   
   # Update changelog
   # Edit CHANGELOG.md
   ```

2. **Create Release PR**
   ```bash
   git checkout -b release/1.2.3
   git commit -m "chore: bump version to 1.2.3"
   git push origin release/1.2.3
   
   # Create PR with release notes
   ```

3. **Deploy and Tag**
   ```bash
   # After PR merge
   git checkout main
   git pull origin main
   git tag v1.2.3
   git push origin v1.2.3
   ```

#### Hotfix Process

For critical production issues:

```bash
# Create hotfix branch from main
git checkout -b hotfix/critical-security-fix main

# Make minimal fix
git commit -m "fix(security): address critical vulnerability

Fixes #789"

# Create emergency PR
# Deploy immediately after review
```

## Quality Assurance

### Definition of Done

A feature is considered "done" when:

- [ ] **Functionality**
  - [ ] Requirements met
  - [ ] Edge cases handled
  - [ ] Error handling implemented

- [ ] **Code Quality**
  - [ ] Code review approved
  - [ ] Standards compliance verified
  - [ ] Performance requirements met

- [ ] **Testing**
  - [ ] Unit tests written and passing
  - [ ] Integration tests passing
  - [ ] Manual testing completed

- [ ] **Documentation**
  - [ ] Code documented
  - [ ] API documentation updated
  - [ ] User documentation updated (if needed)

- [ ] **Deployment**
  - [ ] CI pipeline passing
  - [ ] Database migrations included
  - [ ] Configuration documented

### Testing Strategy

#### Test Pyramid

```
    /\
   /  \     E2E Tests (Few)
  /____\    
 /      \   Integration Tests (Some)
/__________\ Unit Tests (Many)
```

**Unit Tests** (70%):
- Test individual functions
- Mock external dependencies
- Fast execution (<1s total)
- High coverage (>85%)

**Integration Tests** (20%):
- Test component interactions
- Real database connections
- API endpoint testing
- Moderate execution time

**End-to-End Tests** (10%):
- Full user workflows
- Browser automation (if applicable)
- Production-like environment
- Slower execution

#### Test Organization

```
tests/
├── unit/
│   ├── test_validators.py
│   ├── test_rag_engine.py
│   └── test_config.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   └── test_rag_workflow.py
└── e2e/
    └── test_user_workflows.py
```

#### Test Standards

**Test Naming**:
```python
def test_validate_query_when_valid_input_should_return_clean_query():
    """Test query validation with valid input returns cleaned query."""
```

**Test Structure** (Arrange-Act-Assert):
```python
def test_rate_limiter_when_limit_exceeded_should_reject_request():
    # Arrange
    ip_address = "192.168.1.1"
    rate_limiter = RateLimiter()
    
    # Exceed rate limit
    for _ in range(Config.RATE_LIMIT_PER_MINUTE + 1):
        rate_limiter.check_rate_limit(ip_address)
    
    # Act
    result = rate_limiter.check_rate_limit(ip_address)
    
    # Assert
    assert result is False
```

### Performance Standards

#### Response Time Targets

| Endpoint | Target | Maximum |
|----------|--------|---------|
| GET / | <200ms | 500ms |
| POST /api/chat | <2s | 5s |
| GET /api/stats | <100ms | 300ms |
| GET /healthz | <50ms | 100ms |

#### Resource Usage Limits

- **Memory**: <500MB per worker process
- **Database Connections**: <10 per worker
- **OpenAI API**: <100 requests/minute
- **Disk Usage**: <1GB for cache and logs

#### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(endpoint_name: str):
    """Decorator to monitor endpoint performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log performance metrics
                logger.info(f"Performance: {endpoint_name} took {duration:.3f}s")
                
                # Alert if too slow
                if duration > 5.0:
                    logger.warning(f"Slow response: {endpoint_name} took {duration:.3f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Error in {endpoint_name} after {duration:.3f}s: {e}")
                raise
        return wrapper
    return decorator

@monitor_performance("api_chat")
def api_chat():
    # Endpoint implementation
    pass
```

### Documentation Standards

#### Code Documentation

**Module docstrings**:
```python
"""
RAG Engine implementation for document retrieval and response generation.

This module handles the core RAG functionality including:
- Vector storage and retrieval using ChromaDB
- OpenAI integration for embeddings and completions
- Caching for performance optimization
- Error handling and retry logic
"""
```

**Function docstrings**:
```python
def validate_query(query: str, max_length: int = 500) -> Dict[str, Any]:
    """
    Validate and sanitize user query input.
    
    Performs security validation including XSS prevention, SQL injection
    detection, and content sanitization. Returns validation results
    with cleaned query if valid.
    
    Args:
        query: Raw user input query string
        max_length: Maximum allowed query length in characters
        
    Returns:
        Dictionary containing:
            - valid (bool): Whether query passed validation
            - query (str): Sanitized query string if valid
            - errors (List[str]): Validation error messages if invalid
            - warnings (List[str]): Non-blocking warnings
            
    Example:
        >>> result = validate_query("What is React?")
        >>> result['valid']
        True
        >>> result['query']
        'What is React?'
    """
```

#### API Documentation

Keep API documentation up-to-date:

```python
@main_bp.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Process chat query and return AI response.
    
    Request Body:
        {
            "query": "string",  // User question (required)
            "context": "string" // Additional context (optional)
        }
    
    Response:
        {
            "response": "string",      // AI generated response
            "sources": ["string"],     // Source documentation URLs
            "response_time": float,    // Processing time in seconds
            "session_id": "string"     // Session identifier
        }
    
    Status Codes:
        200: Success
        400: Invalid request (missing query, malformed JSON)
        429: Rate limit exceeded
        500: Internal server error
    """
```

---

*These engineering practices ensure consistent, high-quality development while maintaining security and performance standards. Review and update these practices regularly as the team and project evolve.*