import time
import logging
from datetime import datetime, timedelta
from flask import request, jsonify
from functools import wraps
from models import RateLimit
from app import db
from config import Config

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self):
        self.rate_limit = Config.RATE_LIMIT_PER_MINUTE
        self.window_size = 60  # 1 minute in seconds
    
    def is_rate_limited(self, ip_address: str) -> tuple[bool, int]:
        """
        Check if IP is rate limited
        Returns: (is_limited, requests_remaining)
        """
        try:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(seconds=self.window_size)
            
            # Get or create rate limit record
            rate_record = RateLimit.query.filter_by(ip_address=ip_address).first()
            
            if not rate_record:
                # Create new record
                rate_record = RateLimit(
                    ip_address=ip_address,
                    request_count=1,
                    last_request=current_time,
                    reset_time=current_time + timedelta(seconds=self.window_size)
                )
                db.session.add(rate_record)
                db.session.commit()
                
                logger.debug(f"Created new rate limit record for IP: {ip_address}")
                return False, self.rate_limit - 1
            
            # Check if window has reset
            if current_time >= rate_record.reset_time:
                # Reset the window
                rate_record.request_count = 1
                rate_record.last_request = current_time
                rate_record.reset_time = current_time + timedelta(seconds=self.window_size)
                db.session.commit()
                
                logger.debug(f"Reset rate limit window for IP: {ip_address}")
                return False, self.rate_limit - 1
            
            # Check if limit exceeded
            if rate_record.request_count >= self.rate_limit:
                logger.warning(f"Rate limit exceeded for IP: {ip_address}")
                return True, 0
            
            # Increment request count
            rate_record.request_count += 1
            rate_record.last_request = current_time
            db.session.commit()
            
            remaining = self.rate_limit - rate_record.request_count
            logger.debug(f"Rate limit check for IP {ip_address}: {rate_record.request_count}/{self.rate_limit}")
            
            return False, remaining
            
        except Exception as e:
            logger.error(f"Error checking rate limit for IP {ip_address}: {e}")
            # On error, allow the request to proceed
            return False, self.rate_limit
    
    def get_reset_time(self, ip_address: str) -> datetime:
        """Get the reset time for an IP address"""
        try:
            rate_record = RateLimit.query.filter_by(ip_address=ip_address).first()
            if rate_record:
                return rate_record.reset_time
            return datetime.utcnow()
        except Exception as e:
            logger.error(f"Error getting reset time for IP {ip_address}: {e}")
            return datetime.utcnow()
    
    def cleanup_old_records(self) -> int:
        """Clean up old rate limit records"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            old_records = RateLimit.query.filter(RateLimit.last_request < cutoff_time).all()
            
            count = len(old_records)
            for record in old_records:
                db.session.delete(record)
            
            db.session.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} old rate limit records")
            
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up rate limit records: {e}")
            return 0

# Rate limiter instance
rate_limiter = RateLimiter()

def rate_limit_decorator(f):
    """Decorator to apply rate limiting to routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if ip_address:
            ip_address = ip_address.split(',')[0].strip()
        else:
            ip_address = request.remote_addr or 'unknown'
        
        # Check rate limit
        is_limited, remaining = rate_limiter.is_rate_limited(ip_address)
        
        if is_limited:
            reset_time = rate_limiter.get_reset_time(ip_address)
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': f'Too many requests. Limit: {Config.RATE_LIMIT_PER_MINUTE} requests per minute',
                'retry_after': int((reset_time - datetime.utcnow()).total_seconds()),
                'limit': Config.RATE_LIMIT_PER_MINUTE,
                'remaining': 0
            }), 429
        
        # Add rate limit headers to response
        response = f(*args, **kwargs)
        
        if hasattr(response, 'headers'):
            response.headers['X-RateLimit-Limit'] = str(Config.RATE_LIMIT_PER_MINUTE)
            response.headers['X-RateLimit-Remaining'] = str(remaining)
            response.headers['X-RateLimit-Reset'] = str(int(rate_limiter.get_reset_time(ip_address).timestamp()))
        
        return response
    
    return decorated_function
