"""
Async Cache Manager with aiofiles

Non-blocking file I/O for high-performance caching
"""

import json
import time
import logging
from typing import Any, Optional
import os
import aiofiles
import asyncio
from config import Config

logger = logging.getLogger(__name__)


class AsyncCacheManager:
    """Async cache manager with file-based storage"""

    def __init__(self):
        """Initialize async cache manager"""
        self.cache_dir = os.path.join(os.getcwd(), 'cache')
        self.cache_ttl = Config.CACHE_TTL

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(f"Async cache manager initialized with TTL: {self.cache_ttl}s")

    def _get_cache_file_path(self, key: str) -> str:
        """Get the file path for a cache key"""
        safe_key = str(hash(key))
        return os.path.join(self.cache_dir, f"{safe_key}.cache")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (async)"""
        try:
            cache_file = self._get_cache_file_path(key)

            if not os.path.exists(cache_file):
                return None

            # Async file read
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                cache_data = json.loads(content)

            # Check if expired
            if time.time() > cache_data['expires_at']:
                await asyncio.to_thread(os.remove, cache_file)
                return None

            logger.debug(f"Cache hit for key: {key[:50]}...")
            return cache_data['value']

        except Exception as e:
            logger.error(f"Error getting cache for key {key[:50]}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (async)"""
        try:
            cache_file = self._get_cache_file_path(key)
            ttl = ttl or self.cache_ttl

            cache_data = {
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + ttl
            }

            # Async file write
            cache_json = json.dumps(cache_data, ensure_ascii=False, indent=2)
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(cache_json)

            logger.debug(f"Cache set for key: {key[:50]}... (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Error setting cache for key {key[:50]}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache (async)"""
        try:
            cache_file = self._get_cache_file_path(key)

            if os.path.exists(cache_file):
                await asyncio.to_thread(os.remove, cache_file)
                logger.debug(f"Cache deleted for key: {key[:50]}...")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting cache for key {key[:50]}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache (async)"""
        try:
            # Use to_thread for directory operations
            def clear_cache_sync():
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.cache_dir, filename))

            await asyncio.to_thread(clear_cache_sync)
            logger.info("Cache cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def cleanup_expired(self) -> int:
        """Remove expired cache entries (async)"""
        removed_count = 0

        try:
            current_time = time.time()

            # List files in to_thread
            cache_files = await asyncio.to_thread(
                lambda: [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
            )

            for filename in cache_files:
                cache_file = os.path.join(self.cache_dir, filename)

                try:
                    # Async file read
                    async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        cache_data = json.loads(content)

                    if current_time > cache_data['expires_at']:
                        await asyncio.to_thread(os.remove, cache_file)
                        removed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    await asyncio.to_thread(os.remove, cache_file)
                    removed_count += 1

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired cache entries")

            return removed_count

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get cache statistics (async)"""
        try:
            # List files in to_thread
            cache_files = await asyncio.to_thread(
                lambda: [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
            )

            total_files = len(cache_files)
            total_size = 0
            expired_count = 0
            current_time = time.time()

            for filename in cache_files:
                cache_file = os.path.join(self.cache_dir, filename)

                # Get file size in to_thread
                file_size = await asyncio.to_thread(os.path.getsize, cache_file)
                total_size += file_size

                try:
                    async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        cache_data = json.loads(content)

                    if current_time > cache_data['expires_at']:
                        expired_count += 1

                except Exception:
                    expired_count += 1

            return {
                'total_entries': total_files,
                'expired_entries': expired_count,
                'active_entries': total_files - expired_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': self.cache_dir
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'total_entries': 0,
                'expired_entries': 0,
                'active_entries': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0.0,
                'cache_dir': self.cache_dir
            }
