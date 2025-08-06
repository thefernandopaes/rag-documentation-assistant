import json
import time
import logging
from typing import Any, Optional
import os
from config import Config

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        """Initialize cache manager with file-based cache"""
        self.cache_dir = os.path.join(os.getcwd(), 'cache')
        self.cache_ttl = Config.CACHE_TTL
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Cache manager initialized with TTL: {self.cache_ttl}s")
    
    def _get_cache_file_path(self, key: str) -> str:
        """Get the file path for a cache key"""
        # Create a safe filename from the key
        safe_key = str(hash(key))
        return os.path.join(self.cache_dir, f"{safe_key}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            if time.time() > cache_data['expires_at']:
                os.remove(cache_file)
                return None
            
            logger.debug(f"Cache hit for key: {key[:50]}...")
            return cache_data['value']
            
        except Exception as e:
            logger.error(f"Error getting cache for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            cache_file = self._get_cache_file_path(key)
            ttl = ttl or self.cache_ttl
            
            cache_data = {
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + ttl
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Cache set for key: {key[:50]}... (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.debug(f"Cache deleted for key: {key[:50]}...")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    os.remove(os.path.join(self.cache_dir, filename))
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries"""
        removed_count = 0
        
        try:
            current_time = time.time()
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.cache'):
                    continue
                
                cache_file = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if current_time > cache_data['expires_at']:
                        os.remove(cache_file)
                        removed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing cache file {filename}: {e}")
                    # Remove corrupted cache files
                    os.remove(cache_file)
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired cache entries")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
            total_files = len(cache_files)
            total_size = 0
            expired_count = 0
            current_time = time.time()
            
            for filename in cache_files:
                cache_file = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(cache_file)
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if current_time > cache_data['expires_at']:
                        expired_count += 1
                        
                except Exception:
                    expired_count += 1
            
            return {
                'total_entries': total_files,
                'expired_entries': expired_count,
                'active_entries': total_files - expired_count,
                'total_size_bytes': total_size,
                'cache_dir': self.cache_dir
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'total_entries': 0,
                'expired_entries': 0,
                'active_entries': 0,
                'total_size_bytes': 0,
                'cache_dir': self.cache_dir
            }
