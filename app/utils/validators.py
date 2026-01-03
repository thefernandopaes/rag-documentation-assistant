"""
Input validation utilities for DocRag system
Handles query validation, input sanitization, and security checks
"""

import re
import logging
from typing import Optional, Dict, Any, List
import html
import unicodedata

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, code: Optional[str] = None):
        self.message = message
        self.code = code
        super().__init__(self.message)

def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> Dict[str, Any]:
    """
    Validate user query input
    Returns validation result with cleaned query
    """
    result = {
        'valid': False,
        'query': '',
        'errors': [],
        'warnings': []
    }
    
    # Check if query exists
    if not query:
        result['errors'].append('Query cannot be empty')
        return result
    
    # Check type
    if not isinstance(query, str):
        result['errors'].append('Query must be a string')
        return result
    
    # Clean and normalize
    cleaned_query = sanitize_input(query)
    
    # Length validation
    if len(cleaned_query) < min_length:
        result['errors'].append(f'Query must be at least {min_length} characters long')
        return result
    
    if len(cleaned_query) > max_length:
        result['errors'].append(f'Query cannot exceed {max_length} characters')
        return result
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript protocol
        r'on\w+\s*=',      # Event handlers
        r'eval\s*\(',      # eval() calls
        r'document\.',     # Document object access
        r'window\.',       # Window object access
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            result['errors'].append('Query contains potentially malicious content')
            return result
    
    # Check for SQL injection patterns
    sql_patterns = [
        r'\b(union|select|insert|update|delete|drop|alter|create)\b\s+',
        r'--\s*',          # SQL comments
        r'/\*.*\*/',       # SQL block comments
        r';\s*$',          # Semicolon at end
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            result['warnings'].append('Query might contain SQL-like syntax')
            break
    
    # Check for excessive special characters
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', cleaned_query)) / len(cleaned_query)
    if special_char_ratio > 0.3:
        result['warnings'].append('Query contains many special characters')
    
    # Check for repeated characters (potential spam)
    if re.search(r'(.)\1{10,}', cleaned_query):
        result['errors'].append('Query contains excessive repeated characters')
        return result
    
    # All validations passed
    result['valid'] = True
    result['query'] = cleaned_query
    
    return result

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format
    This is a basic validation - real implementation would check against database
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format validation (adjust based on your API key format)
    # OpenAI keys typically start with 'sk-' and are 48+ characters
    if api_key.startswith('sk-') and len(api_key) >= 48:
        return True
    
    # Generic API key pattern
    if re.match(r'^[a-zA-Z0-9\-_]{20,}$', api_key):
        return True
    
    return False

def sanitize_input(text: str, preserve_newlines: bool = False) -> str:
    """
    Sanitize user input to prevent XSS and other injection attacks
    """
    if not text:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # HTML escape
    text = html.escape(text)
    
    # Remove control characters except newlines and tabs
    if preserve_newlines:
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    else:
        text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def validate_email(email: str) -> bool:
    """Validate email format"""
    if not email or not isinstance(email, str):
        return False
    
    # Basic email regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_username(username: str, min_length: int = 3, max_length: int = 30) -> Dict[str, Any]:
    """Validate username format"""
    result = {
        'valid': False,
        'errors': []
    }
    
    if not username:
        result['errors'].append('Username cannot be empty')
        return result
    
    if not isinstance(username, str):
        result['errors'].append('Username must be a string')
        return result
    
    if len(username) < min_length:
        result['errors'].append(f'Username must be at least {min_length} characters')
        return result
    
    if len(username) > max_length:
        result['errors'].append(f'Username cannot exceed {max_length} characters')
        return result
    
    # Allow alphanumeric, underscore, dash
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        result['errors'].append('Username can only contain letters, numbers, underscore, and dash')
        return result
    
    # Must start with letter or number
    if not re.match(r'^[a-zA-Z0-9]', username):
        result['errors'].append('Username must start with a letter or number')
        return result
    
    result['valid'] = True
    return result

def validate_url(url: str) -> bool:
    """Validate URL format"""
    if not url or not isinstance(url, str):
        return False
    
    # Basic URL pattern
    pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$'
    return bool(re.match(pattern, url))

def check_rate_limit_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """Extract rate limit information from headers"""
    rate_limit_info = {
        'limit': None,
        'remaining': None,
        'reset': None,
        'retry_after': None
    }
    
    # Common rate limit header names
    header_mappings = {
        'x-ratelimit-limit': 'limit',
        'x-ratelimit-remaining': 'remaining',
        'x-ratelimit-reset': 'reset',
        'retry-after': 'retry_after'
    }
    
    for header_name, info_key in header_mappings.items():
        value = headers.get(header_name) or headers.get(header_name.upper())
        if value:
            try:
                if value is not None:
                    rate_limit_info[info_key] = int(value)
            except ValueError:
                logger.warning(f"Invalid rate limit header value: {header_name}={value}")
    
    return rate_limit_info

def validate_file_upload(filename: str, content_type: str, max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
    """Validate file upload parameters"""
    result = {
        'valid': False,
        'errors': []
    }
    
    if not filename:
        result['errors'].append('Filename is required')
        return result
    
    # Check file extension
    allowed_extensions = ['.txt', '.md', '.py', '.js', '.json', '.html', '.css']
    file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    
    if file_ext not in allowed_extensions:
        result['errors'].append(f'File type not allowed. Allowed types: {", ".join(allowed_extensions)}')
        return result
    
    # Check content type
    allowed_content_types = [
        'text/plain',
        'text/markdown',
        'text/x-python',
        'application/javascript',
        'application/json',
        'text/html',
        'text/css'
    ]
    
    if content_type not in allowed_content_types:
        result['errors'].append('Content type not allowed')
        return result
    
    result['valid'] = True
    return result

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    if not filename:
        return "untitled"
    
    # Remove path separators and other dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[\x00-\x1F\x7F]', '', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:251-len(ext)] + ('.' + ext if ext else '')
    
    return filename or "untitled"

def validate_json_structure(data: Any, required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate JSON data structure"""
    result = {
        'valid': False,
        'errors': []
    }
    
    if not isinstance(data, dict):
        result['errors'].append('Data must be a JSON object')
        return result
    
    if required_fields:
        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            result['errors'].append(f'Missing required fields: {", ".join(missing_fields)}')
            return result
    
    result['valid'] = True
    return result
