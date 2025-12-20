"""
Enhanced Security Module for API Documentation RAG System.

This module provides comprehensive security features specifically designed
for protecting API documentation assistance systems, including request validation,
API key management, and threat protection.
"""

import hashlib
import hmac
import time
import secrets
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlparse
import re

from flask import request, jsonify, current_app, session
from werkzeug.exceptions import TooManyRequests, BadRequest, Forbidden
from config import Config

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration and constants."""
    
    # Request size limits
    MAX_QUERY_LENGTH = 2000
    MAX_URL_LENGTH = 500
    MAX_HEADERS = 50
    
    # Rate limiting windows
    RATE_LIMIT_WINDOWS = {
        'minute': 60,
        'hour': 3600,
        'day': 86400
    }
    
    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS attempts
        r'javascript:',  # JavaScript injection
        r'data:(?!image)',  # Data URI (except images)
        r'file://',  # File URI
        r'ftp://',  # FTP URI
        r'\.\./|\.\.\\'  # Path traversal
    ]
    
    # Allowed domains for API discovery
    TRUSTED_API_DOMAINS = {
        'github.com', 'api.github.com',
        'stripe.com', 'api.stripe.com',
        'openai.com', 'api.openai.com',
        'twilio.com', 'api.twilio.com',
        'discord.com', 'discord.com/api',
        'googleapis.com',
        'jsonplaceholder.typicode.com',
        'httpbin.org',
        'postman-echo.com'
    }


class SecurityValidator:
    """Comprehensive security validation for API documentation requests."""
    
    def __init__(self):
        self.request_history = {}  # For tracking request patterns
        self.blocked_ips = set()
        self.api_key_usage = {}
        
    def validate_request(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive request validation.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic request structure validation
            if not self._validate_request_structure(request_data):
                return False, "Invalid request structure"
            
            # Query content validation
            if not self._validate_query_content(request_data.get('query', '')):
                return False, "Invalid query content detected"
            
            # URL validation (if present)
            if 'url' in request_data and not self._validate_url(request_data['url']):
                return False, "Invalid or untrusted URL"
            
            # Rate limiting check
            if not self._check_rate_limits(request.remote_addr):
                return False, "Rate limit exceeded"
            
            # Check for suspicious patterns
            if self._detect_suspicious_patterns(request_data):
                return False, "Suspicious content detected"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error during request validation: {e}")
            return False, "Validation error"
    
    def validate_api_discovery_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate URLs for API discovery to prevent SSRF attacks.
        
        Args:
            url: The URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic URL format validation
            if not self._validate_url_format(url):
                return False, "Invalid URL format"
            
            # Parse URL
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False, "Only HTTP/HTTPS URLs are allowed"
            
            # Enforce HTTPS for production
            if Config._is_production() and parsed.scheme != 'https':
                return False, "HTTPS required in production"
            
            # Check domain whitelist
            if Config.ALLOWED_DOMAINS and parsed.netloc not in Config.ALLOWED_DOMAINS:
                # Check against trusted API domains
                if not any(trusted_domain in parsed.netloc for trusted_domain in SecurityConfig.TRUSTED_API_DOMAINS):
                    return False, f"Domain not in allowed list: {parsed.netloc}"
            
            # Prevent private IP access (SSRF protection)
            if self._is_private_ip(parsed.hostname):
                return False, "Access to private IPs not allowed"
            
            # Check for suspicious URL patterns
            if self._has_suspicious_url_patterns(url):
                return False, "Suspicious URL pattern detected"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating API discovery URL: {e}")
            return False, "URL validation error"
    
    def sanitize_query(self, query: str) -> str:
        """
        Sanitize user query to prevent injection attacks.
        
        Args:
            query: Raw user query
            
        Returns:
            Sanitized query string
        """
        try:
            # Remove HTML tags
            query = re.sub(r'<[^>]+>', '', query)
            
            # Remove suspicious patterns
            for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
                query = re.sub(pattern, '', query, flags=re.IGNORECASE)
            
            # Limit length
            if len(query) > SecurityConfig.MAX_QUERY_LENGTH:
                query = query[:SecurityConfig.MAX_QUERY_LENGTH]
            
            # Remove excessive whitespace
            query = re.sub(r'\s+', ' ', query).strip()
            
            return query
            
        except Exception as e:
            logger.error(f"Error sanitizing query: {e}")
            return ""
    
    def generate_secure_session_id(self) -> str:
        """Generate a cryptographically secure session ID."""
        return secrets.token_urlsafe(32)
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session ID format and security."""
        if not session_id:
            return False
        
        # Check length and format
        if len(session_id) < 20 or not re.match(r'^[A-Za-z0-9_-]+$', session_id):
            return False
        
        return True
    
    def track_api_usage(self, api_key: Optional[str], endpoint: str):
        """Track API key usage for monitoring and rate limiting."""
        if not api_key:
            return
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        current_time = datetime.utcnow()
        
        if key_hash not in self.api_key_usage:
            self.api_key_usage[key_hash] = {
                'requests': [],
                'endpoints': set(),
                'first_seen': current_time,
                'last_seen': current_time
            }
        
        usage_data = self.api_key_usage[key_hash]
        usage_data['requests'].append(current_time)
        usage_data['endpoints'].add(endpoint)
        usage_data['last_seen'] = current_time
        
        # Clean old requests (keep last 24 hours)
        cutoff = current_time - timedelta(hours=24)
        usage_data['requests'] = [
            req_time for req_time in usage_data['requests']
            if req_time > cutoff
        ]
    
    def _validate_request_structure(self, request_data: Dict[str, Any]) -> bool:
        """Validate basic request structure."""
        # Check required fields
        if 'query' not in request_data:
            return False
        
        # Check data types
        if not isinstance(request_data['query'], str):
            return False
        
        # Check for excessive nesting or data
        if self._is_excessive_data(request_data):
            return False
        
        return True
    
    def _validate_query_content(self, query: str) -> bool:
        """Validate query content for security."""
        # Length check
        if len(query) > SecurityConfig.MAX_QUERY_LENGTH:
            return False
        
        # Check for null bytes
        if '\x00' in query:
            return False
        
        # Check for control characters (except common whitespace)
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', query):
            return False
        
        return True
    
    def _validate_url(self, url: str) -> bool:
        """Basic URL validation."""
        return self.validate_api_discovery_url(url)[0]
    
    def _validate_url_format(self, url: str) -> bool:
        """Validate URL format."""
        if not url or len(url) > SecurityConfig.MAX_URL_LENGTH:
            return False
        
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def _is_private_ip(self, hostname: Optional[str]) -> bool:
        """Check if hostname resolves to a private IP."""
        if not hostname:
            return False
        
        import socket
        import ipaddress
        
        try:
            # Resolve hostname to IP
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            
            # Check if it's a private IP
            return ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
            
        except (socket.gaierror, ipaddress.AddressValueError):
            # If we can't resolve, be safe and block
            return True
    
    def _has_suspicious_url_patterns(self, url: str) -> bool:
        """Check for suspicious URL patterns."""
        suspicious_patterns = [
            r'localhost',
            r'127\.0\.0\.1',
            r'0\.0\.0\.0',
            r'169\.254\.',  # Link-local
            r'10\.',        # Private Class A
            r'172\.(1[6-9]|2[0-9]|3[01])\.',  # Private Class B
            r'192\.168\.',  # Private Class C
            r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+',  # IP with port
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def _check_rate_limits(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limits."""
        current_time = time.time()
        
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        
        # Clean old requests
        cutoff = current_time - SecurityConfig.RATE_LIMIT_WINDOWS['minute']
        self.request_history[client_ip] = [
            req_time for req_time in self.request_history[client_ip]
            if req_time > cutoff
        ]
        
        # Check current rate
        request_count = len(self.request_history[client_ip])
        
        if request_count >= Config.RATE_LIMIT_PER_MINUTE:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False
        
        # Add current request
        self.request_history[client_ip].append(current_time)
        return True
    
    def _detect_suspicious_patterns(self, request_data: Dict[str, Any]) -> bool:
        """Detect suspicious patterns in request data."""
        query = request_data.get('query', '')
        
        for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return True
        
        return False
    
    def _is_excessive_data(self, data: Any, max_depth: int = 5, current_depth: int = 0) -> bool:
        """Check if data structure is excessively deep or large."""
        if current_depth > max_depth:
            return True
        
        if isinstance(data, dict):
            if len(data) > 100:  # Too many keys
                return True
            for value in data.values():
                if self._is_excessive_data(value, max_depth, current_depth + 1):
                    return True
        
        elif isinstance(data, list):
            if len(data) > 1000:  # Too many items
                return True
            for item in data:
                if self._is_excessive_data(item, max_depth, current_depth + 1):
                    return True
        
        elif isinstance(data, str):
            if len(data) > 10000:  # Too long string
                return True
        
        return False


class SecurityMiddleware:
    """Security middleware for Flask application."""
    
    def __init__(self, app=None, validator=None):
        self.app = app
        self.validator = validator or SecurityValidator()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security middleware with Flask app."""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Register error handlers
        app.errorhandler(TooManyRequests)(self.handle_rate_limit)
        app.errorhandler(BadRequest)(self.handle_bad_request)
        app.errorhandler(Forbidden)(self.handle_forbidden)
    
    def before_request(self):
        """Run security checks before each request."""
        try:
            # Skip security checks for static files
            if request.endpoint and request.endpoint.startswith('static'):
                return
            
            # Check if IP is blocked
            if request.remote_addr in self.validator.blocked_ips:
                logger.warning(f"Blocked IP attempted access: {request.remote_addr}")
                return jsonify({'error': 'Access denied'}), 403
            
            # Validate headers
            if not self._validate_headers():
                return jsonify({'error': 'Invalid headers'}), 400
            
            # For API endpoints, validate request data
            if request.endpoint and request.endpoint.startswith('main.api_'):
                if request.is_json and request.get_json():
                    is_valid, error = self.validator.validate_request(request.get_json())
                    if not is_valid:
                        logger.warning(f"Invalid request from {request.remote_addr}: {error}")
                        return jsonify({'error': error}), 400
            
        except Exception as e:
            logger.error(f"Error in security middleware: {e}")
            return jsonify({'error': 'Security validation failed'}), 500
    
    def after_request(self, response):
        """Add security headers after each request."""
        try:
            # Security headers
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
            
            # CSP for production
            if Config._is_production():
                csp = (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com cdn.jsdelivr.net; "
                    "style-src 'self' 'unsafe-inline' cdnjs.cloudflare.com cdn.replit.com; "
                    "font-src 'self' cdnjs.cloudflare.com; "
                    "img-src 'self' data:; "
                    "connect-src 'self';"
                )
                response.headers['Content-Security-Policy'] = csp
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding security headers: {e}")
            return response
    
    def _validate_headers(self) -> bool:
        """Validate request headers."""
        # Check header count
        if len(request.headers) > SecurityConfig.MAX_HEADERS:
            return False
        
        # Check for suspicious headers
        for header_name, header_value in request.headers:
            if len(header_name) > 100 or len(header_value) > 1000:
                return False
            
            # Check for injection attempts in headers
            for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
                if re.search(pattern, header_value, re.IGNORECASE):
                    return False
        
        return True
    
    def handle_rate_limit(self, error):
        """Handle rate limit errors."""
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.',
            'retry_after': 60
        }), 429
    
    def handle_bad_request(self, error):
        """Handle bad request errors."""
        return jsonify({
            'error': 'Bad request',
            'message': 'Invalid request format or content'
        }), 400
    
    def handle_forbidden(self, error):
        """Handle forbidden errors."""
        return jsonify({
            'error': 'Forbidden',
            'message': 'Access denied'
        }), 403


def secure_api_endpoint(f):
    """Decorator for additional API endpoint security."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Additional validation for API endpoints
            validator = SecurityValidator()
            
            # Check session validity
            session_id = session.get('session_id')
            if session_id and not validator.validate_session(session_id):
                session.clear()
                return jsonify({'error': 'Invalid session'}), 401
            
            # Sanitize query if present
            if request.is_json:
                data = request.get_json()
                if data and 'query' in data:
                    data['query'] = validator.sanitize_query(data['query'])
                    request._cached_json = (data, None)
            
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in secure_api_endpoint decorator: {e}")
            return jsonify({'error': 'Security validation failed'}), 500
    
    return decorated_function


def init_security(app):
    """Initialize security system with Flask app."""
    validator = SecurityValidator()
    middleware = SecurityMiddleware(app, validator)
    
    logger.info("Security system initialized")
    return validator, middleware