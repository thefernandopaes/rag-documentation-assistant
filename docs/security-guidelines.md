# Security Guidelines

This document outlines security practices, secret management, and vulnerability prevention measures for the DocRag application.

## Security Architecture Overview

The DocRag application implements defense-in-depth security practices across multiple layers:

1. **Input Layer**: Validation and sanitization
2. **Application Layer**: Authentication and authorization
3. **Data Layer**: Secure storage and transmission
4. **Infrastructure Layer**: Network and deployment security

## Input Security

### Request Validation

All user inputs are validated using the `utils/validators.py` module:

```python
from utils.validators import validate_query

# Validate user query
validation_result = validate_query(user_input)
if not validation_result['valid']:
    return jsonify({'error': validation_result['errors']}), 400

clean_query = validation_result['query']
```

### XSS Prevention

**HTML Escaping**: All user input is HTML-escaped automatically:

```python
import html
from utils.validators import sanitize_input

# Automatic HTML escaping in sanitize_input
def sanitize_input(text: str) -> str:
    text = html.escape(text)  # Escape HTML entities
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)  # Remove control characters
    return text.strip()
```

**Content Security Policy**: Implement CSP headers in production:

```python
@app.after_request
def add_security_headers(response):
    if Config._is_production():
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'"
        )
    return response
```

### SQL Injection Prevention

**Parameterized Queries**: Always use SQLAlchemy ORM or parameterized queries:

```python
# Good: Using ORM
conversation = Conversation.query.filter_by(session_id=session_id).all()

# Good: Parameterized raw query
result = db.session.execute(
    text("SELECT * FROM conversation WHERE session_id = :session_id"),
    {"session_id": session_id}
)

# Bad: String concatenation (NEVER DO THIS)
# query = f"SELECT * FROM conversation WHERE session_id = '{session_id}'"
```

### Malicious Content Detection

The validator detects suspicious patterns:

```python
suspicious_patterns = [
    r'<script[^>]*>',  # Script tags
    r'javascript:',     # JavaScript protocol
    r'on\w+\s*=',      # Event handlers
    r'eval\s*\(',      # eval() calls
]

for pattern in suspicious_patterns:
    if re.search(pattern, cleaned_query, re.IGNORECASE):
        result['errors'].append('Query contains potentially malicious content')
        return result
```

## Authentication & Authorization

### API Key Protection

**Admin Endpoints**: Protected by API key authentication:

```python
def validate_admin_key(auth_header: str) -> bool:
    """Validate admin API key from Authorization header."""
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
    
    provided_key = auth_header[7:]  # Remove 'Bearer ' prefix
    expected_key = Config.ADMIN_API_KEY
    
    if not expected_key:
        return False
    
    # Use constant-time comparison to prevent timing attacks
    import hmac
    return hmac.compare_digest(provided_key, expected_key)

@main_bp.route('/api/initialize', methods=['POST'])
def initialize():
    auth_header = request.headers.get('Authorization')
    if not validate_admin_key(auth_header):
        return jsonify({'error': 'Unauthorized'}), 401
```

### Session Management

**Secure Session Configuration**:

```python
if Config._is_production():
    app.config.update(
        SESSION_COOKIE_SECURE=True,    # HTTPS only
        SESSION_COOKIE_HTTPONLY=True,  # No JavaScript access
        SESSION_COOKIE_SAMESITE='Lax', # CSRF protection
    )
```

**Session ID Generation**:

```python
import uuid

# Generate cryptographically secure session IDs
session_id = str(uuid.uuid4())
```

## Rate Limiting & Abuse Prevention

### IP-Based Rate Limiting

```python
from rate_limiter import rate_limit_decorator

@main_bp.route('/api/chat', methods=['POST'])
@rate_limit_decorator
def api_chat():
    # Rate limited endpoint implementation
    pass
```

### Rate Limit Implementation

```python
def check_rate_limit(ip_address: str) -> bool:
    """Check if IP address is within rate limits."""
    rate_limit = RateLimit.query.filter_by(ip_address=ip_address).first()
    
    if not rate_limit:
        # First request from this IP
        rate_limit = RateLimit(ip_address=ip_address)
        db.session.add(rate_limit)
    
    now = datetime.utcnow()
    
    # Reset counter if needed
    if now >= rate_limit.reset_time:
        rate_limit.request_count = 0
        rate_limit.reset_time = now + timedelta(minutes=1)
    
    # Check limit
    if rate_limit.request_count >= Config.RATE_LIMIT_PER_MINUTE:
        return False
    
    # Increment counter
    rate_limit.request_count += 1
    rate_limit.last_request = now
    db.session.commit()
    
    return True
```

## Data Security

### Database Security

**Connection Security**: Use SSL/TLS for database connections:

```python
# Production database configuration
engine_options = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

if is_postgres and Config.DB_SSLMODE:
    engine_options["connect_args"] = {"sslmode": Config.DB_SSLMODE}
```

**Data Sanitization**: Sanitize data before storage:

```python
def store_conversation(user_query: str, ai_response: str):
    # Sanitize before storing
    clean_query = sanitize_input(user_query, preserve_newlines=True)
    clean_response = sanitize_input(ai_response, preserve_newlines=True)
    
    conversation = Conversation(
        user_query=clean_query,
        ai_response=clean_response,
        session_id=session['session_id']
    )
    db.session.add(conversation)
    db.session.commit()
```

### Data Encryption

**At Rest**: Use database encryption features:

```sql
-- PostgreSQL: Enable transparent data encryption
ALTER DATABASE docrag SET default_text_search_config = 'pg_catalog.english';
```

**In Transit**: Always use HTTPS/TLS:

```python
# Force HTTPS in production
if Config._is_production():
    app.config['PREFERRED_URL_SCHEME'] = 'https'
```

## Secret Management

### Environment Variables

**Never commit secrets**:

```bash
# .gitignore must include
.env
.env.local
.env.production
secrets/
```

**Secret Validation**: Validate secrets on startup:

```python
@classmethod
def validate_config(cls):
    if not cls.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    if cls._is_production() and not cls.SESSION_SECRET:
        raise ValueError("SESSION_SECRET is required in production")
    
    # Validate secret format
    if cls.SESSION_SECRET and len(cls.SESSION_SECRET) < 32:
        raise ValueError("SESSION_SECRET must be at least 32 characters")
```

### Secret Rotation

**Rotation Strategy**:
1. Generate new secret
2. Update environment variables
3. Deploy new configuration
4. Verify functionality
5. Revoke old secret

**Rotation Script Example**:

```python
#!/usr/bin/env python3
import secrets
import os

def generate_session_secret():
    """Generate a new 64-character hex session secret."""
    return secrets.token_hex(32)

def generate_api_key():
    """Generate a new API key."""
    return secrets.token_urlsafe(32)

if __name__ == "__main__":
    print("New SESSION_SECRET:", generate_session_secret())
    print("New ADMIN_API_KEY:", generate_api_key())
```

### Secret Storage Best Practices

**Development**:
```env
# .env file (never commit)
SESSION_SECRET=your-64-hex-secret
ADMIN_API_KEY=your-admin-key
```

**Production**:
- Use platform-specific secret management (Railway, AWS Secrets Manager, etc.)
- Set environment variables directly in deployment platform
- Use separate secrets for different environments

## OpenAI API Security

### API Key Protection

**Key Validation**:

```python
def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # OpenAI keys start with 'sk-' and are 48+ characters
    if api_key.startswith('sk-') and len(api_key) >= 48:
        return True
    
    return False
```

**Rate Limiting**: Implement OpenAI-specific rate limiting:

```python
class OpenAIRateLimiter:
    def __init__(self):
        self.requests_per_minute = 100
        self.tokens_per_minute = 150000
    
    def check_limits(self, estimated_tokens: int) -> bool:
        # Implement rate limiting logic
        pass
```

### Request Security

**Timeout Configuration**:

```python
client = OpenAI(
    api_key=Config.OPENAI_API_KEY,
    timeout=Config.OPENAI_TIMEOUT,
    max_retries=Config.OPENAI_MAX_RETRIES
)
```

**Error Handling**: Don't expose API details:

```python
try:
    response = client.embeddings.create(...)
except openai.APIError as e:
    logger.error(f"OpenAI API error: {e}")
    # Don't expose API error details to users
    return {"error": "Service temporarily unavailable"}, 503
```

## Web Security Headers

### Security Headers Implementation

```python
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    if Config._is_production():
        response.headers.update({
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
        })
    return response
```

### CORS Configuration

```python
from flask_cors import CORS

# Configure CORS for production
if Config._is_production():
    CORS(app, origins=['https://yourdomain.com'])
else:
    # More permissive for development
    CORS(app, origins=['http://localhost:*'])
```

## File Security

### File Upload Validation

```python
def validate_file_upload(filename: str, content_type: str) -> Dict[str, Any]:
    """Validate file upload security."""
    result = {'valid': False, 'errors': []}
    
    # Check file extension
    allowed_extensions = ['.txt', '.md', '.py', '.js', '.json']
    file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    
    if file_ext not in allowed_extensions:
        result['errors'].append('File type not allowed')
        return result
    
    # Check content type
    allowed_content_types = [
        'text/plain', 'text/markdown', 'application/json'
    ]
    
    if content_type not in allowed_content_types:
        result['errors'].append('Content type not allowed')
        return result
    
    result['valid'] = True
    return result
```

### Filename Sanitization

```python
def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    if not filename:
        return "untitled"
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[\x00-\x1F\x7F]', '', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip(' .')
    
    return filename or "untitled"
```

## Logging Security

### Secure Logging Practices

**Never log sensitive data**:

```python
# Good: Log without sensitive data
logger.info(f"User query processed for session {session_id[:8]}...")

# Bad: Logging sensitive information
# logger.info(f"API key used: {Config.OPENAI_API_KEY}")
# logger.info(f"User query: {user_query}")
```

### Log Sanitization

```python
def sanitize_for_logging(data: str) -> str:
    """Sanitize data before logging."""
    if not data:
        return ""
    
    # Remove potential secrets
    data = re.sub(r'sk-[a-zA-Z0-9]{48,}', '[REDACTED]', data)
    data = re.sub(r'password["\']:\s*["\'][^"\']*["\']', 'password: [REDACTED]', data, re.IGNORECASE)
    
    return data
```

## Vulnerability Management

### Dependency Security

**Regular Updates**: Keep dependencies updated:

```bash
# Check for security vulnerabilities
pip audit

# Update dependencies
pip install --upgrade package-name
```

**Vulnerability Scanning**: Use automated scanning tools:

```yaml
# GitHub Actions security scanning
name: Security Scan
on: [push, pull_request]
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Bandit Security Scan
        run: |
          pip install bandit
          bandit -r . -f json -o bandit-report.json
```

### Security Testing

**Input Validation Tests**:

```python
def test_xss_prevention():
    """Test XSS prevention in query validation."""
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
    ]
    
    for malicious_input in malicious_inputs:
        result = validate_query(malicious_input)
        assert not result['valid']
        assert 'malicious content' in str(result['errors'])
```

**SQL Injection Tests**:

```python
def test_sql_injection_prevention():
    """Test SQL injection prevention."""
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'/**/OR/**/1=1--",
    ]
    
    for malicious_input in malicious_inputs:
        # Should not cause database errors or unauthorized access
        result = validate_query(malicious_input)
        # Should either be rejected or safely handled
        assert not result.get('valid', True)
```

## Security Incident Response

### Detection

**Monitoring Indicators**:
- Multiple failed authentication attempts
- Unusual request patterns
- Rate limit violations
- Suspicious query patterns

**Alerting Setup**:

```python
def detect_security_incident(request_data: dict) -> bool:
    """Detect potential security incidents."""
    red_flags = 0
    
    # Check for suspicious patterns
    if 'user_agent' in request_data:
        if 'scanner' in request_data['user_agent'].lower():
            red_flags += 1
    
    # Check for rapid requests
    if request_data.get('requests_per_minute', 0) > 100:
        red_flags += 1
    
    # Check for malicious payloads
    query = request_data.get('query', '')
    if any(pattern in query.lower() for pattern in ['<script', 'javascript:', 'drop table']):
        red_flags += 1
    
    return red_flags >= 2
```

### Response Procedures

1. **Immediate Response**:
   - Block suspicious IP addresses
   - Revoke compromised API keys
   - Enable additional logging

2. **Investigation**:
   - Review logs for attack patterns
   - Assess potential data exposure
   - Document incident timeline

3. **Recovery**:
   - Patch vulnerabilities
   - Update security measures
   - Communicate with stakeholders

## Security Checklist

### Development

- [ ] Input validation implemented for all user inputs
- [ ] SQL injection prevention (parameterized queries only)
- [ ] XSS prevention (HTML escaping)
- [ ] Rate limiting on all public endpoints
- [ ] Secure session configuration
- [ ] No secrets in version control

### Testing

- [ ] Security tests for input validation
- [ ] XSS prevention tests
- [ ] SQL injection prevention tests
- [ ] Authentication/authorization tests
- [ ] Rate limiting tests

### Deployment

- [ ] HTTPS/TLS enabled
- [ ] Security headers configured
- [ ] Database SSL/TLS enabled
- [ ] Environment variables secured
- [ ] Monitoring and alerting set up
- [ ] Incident response plan documented

### Monitoring

- [ ] Failed authentication attempts logged
- [ ] Rate limit violations monitored
- [ ] Suspicious query patterns detected
- [ ] Security headers validated
- [ ] SSL/TLS certificate monitoring

---

*Security is an ongoing process. Regularly review and update these guidelines as threats evolve and the application grows.*