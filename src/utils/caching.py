"""
Caching system for intermediate results and processed data.
Provides memory and disk-based caching for masks, meshes, and textures.
"""

import hashlib
import pickle
import json
import os
import time
import logging
from typing import Any, Dict, Optional, Union, Callable
from pathlib import Path
import numpy as np
import trimesh
from functools import wraps

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for different types of data."""
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 max_memory_size_mb: int = 512,
                 max_disk_size_mb: int = 2048,
                 default_ttl_seconds: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_size_mb: Maximum memory cache size in MB
            max_disk_size_mb: Maximum disk cache size in MB
            default_ttl_seconds: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_memory_size = max_memory_size_mb * 1024 * 1024  # Convert to bytes
        self.max_disk_size = max_disk_size_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        
        # Memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_size = 0
        
        # Disk cache metadata
        self.disk_metadata_file = self.cache_dir / "metadata.json"
        self.disk_metadata = self._load_disk_metadata()
        
        # Cleanup old entries
        self._cleanup_expired()
    
    def _load_disk_metadata(self) -> Dict[str, Any]:
        """Load disk cache metadata."""
        if self.disk_metadata_file.exists():
            try:
                with open(self.disk_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load disk metadata: {e}")
        return {}
    
    def _save_disk_metadata(self):
        """Save disk cache metadata."""
        try:
            with open(self.disk_metadata_file, 'w') as f:
                json.dump(self.disk_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save disk metadata: {e}")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a hash of the arguments
        key_data = {
            'args': args,
            'kwargs': kwargs,
            'prefix': prefix
        }
        
        # Convert to JSON string and hash
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{prefix}_{key_hash}"
    
    def _get_memory_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, trimesh.Trimesh):
            return data.vertices.nbytes + data.faces.nbytes
        elif isinstance(data, (list, tuple)):
            return sum(self._get_memory_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._get_memory_size(v) for v in data.values())
        else:
            # Rough estimate for other types
            return len(str(data).encode('utf-8'))
    
    def _cleanup_memory(self):
        """Clean up memory cache if it's too large."""
        while self.memory_size > self.max_memory_size and self.memory_cache:
            # Remove oldest entry
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k]['timestamp'])
            self._remove_from_memory(oldest_key)
    
    def _remove_from_memory(self, key: str):
        """Remove entry from memory cache."""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.memory_size -= entry['size']
            del self.memory_cache[key]
    
    def _cleanup_expired(self):
        """Remove expired entries from both caches."""
        current_time = time.time()
        
        # Clean memory cache
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if current_time - entry['timestamp'] > entry['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_from_memory(key)
        
        # Clean disk cache
        expired_disk_keys = []
        for key, metadata in self.disk_metadata.items():
            if current_time - metadata['timestamp'] > metadata['ttl']:
                expired_disk_keys.append(key)
        
        for key in expired_disk_keys:
            self._remove_disk_entry(key)
    
    def _remove_disk_entry(self, key: str):
        """Remove entry from disk cache."""
        if key in self.disk_metadata:
            file_path = self.cache_dir / f"{key}.pkl"
            if file_path.exists():
                file_path.unlink()
            del self.disk_metadata[key]
            self._save_disk_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['timestamp'] <= entry['ttl']:
                return entry['data']
            else:
                self._remove_from_memory(key)
        
        # Try disk cache
        if key in self.disk_metadata:
            metadata = self.disk_metadata[key]
            if time.time() - metadata['timestamp'] <= metadata['ttl']:
                file_path = self.cache_dir / f"{key}.pkl"
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Move to memory cache if there's space
                        data_size = self._get_memory_size(data)
                        if data_size < self.max_memory_size:
                            self._add_to_memory(key, data, metadata['ttl'])
                        
                        return data
                    except Exception as e:
                        logger.warning(f"Could not load from disk cache: {e}")
                        self._remove_disk_entry(key)
            else:
                self._remove_disk_entry(key)
        
        return None
    
    def _add_to_memory(self, key: str, data: Any, ttl: int):
        """Add data to memory cache."""
        data_size = self._get_memory_size(data)
        
        # Check if we have enough space
        if data_size > self.max_memory_size:
            return  # Too large for memory cache
        
        # Clean up if necessary
        while self.memory_size + data_size > self.max_memory_size and self.memory_cache:
            self._cleanup_memory()
        
        # Add to memory cache
        self.memory_cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl,
            'size': data_size
        }
        self.memory_size += data_size
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Put data into cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            # Try to add to memory cache first
            data_size = self._get_memory_size(data)
            if data_size < self.max_memory_size:
                self._add_to_memory(key, data, ttl)
            else:
                # Too large for memory, save to disk
                self._save_to_disk(key, data, ttl)
            
            return True
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            return False
    
    def _save_to_disk(self, key: str, data: Any, ttl: int):
        """Save data to disk cache."""
        file_path = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self.disk_metadata[key] = {
                'timestamp': time.time(),
                'ttl': ttl,
                'file_path': str(file_path),
                'size': file_path.stat().st_size
            }
            self._save_disk_metadata()
            
        except Exception as e:
            logger.error(f"Error saving to disk cache: {e}")
            if file_path.exists():
                file_path.unlink()
    
    def delete(self, key: str):
        """Delete data from cache."""
        self._remove_from_memory(key)
        self._remove_disk_entry(key)
    
    def clear(self):
        """Clear all cache data."""
        self.memory_cache.clear()
        self.memory_size = 0
        
        # Remove all disk cache files
        for key in list(self.disk_metadata.keys()):
            self._remove_disk_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        
        # Memory cache stats
        memory_entries = len(self.memory_cache)
        memory_size_mb = self.memory_size / (1024 * 1024)
        
        # Disk cache stats
        disk_entries = len(self.disk_metadata)
        disk_size = sum(metadata['size'] for metadata in self.disk_metadata.values())
        disk_size_mb = disk_size / (1024 * 1024)
        
        # Expired entries
        expired_memory = sum(1 for entry in self.memory_cache.values() 
                           if current_time - entry['timestamp'] > entry['ttl'])
        expired_disk = sum(1 for metadata in self.disk_metadata.values() 
                         if current_time - metadata['timestamp'] > metadata['ttl'])
        
        return {
            'memory_cache': {
                'entries': memory_entries,
                'size_mb': memory_size_mb,
                'max_size_mb': self.max_memory_size / (1024 * 1024),
                'expired_entries': expired_memory
            },
            'disk_cache': {
                'entries': disk_entries,
                'size_mb': disk_size_mb,
                'max_size_mb': self.max_disk_size / (1024 * 1024),
                'expired_entries': expired_disk
            },
            'total_entries': memory_entries + disk_entries,
            'total_size_mb': memory_size_mb + disk_size_mb
        }


def cached(prefix: str, ttl: Optional[int] = None, cache_type: str = 'auto'):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds
        cache_type: Cache type ('memory', 'disk', 'auto')
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_manager._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.put(key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator


# Global cache manager instance
cache_manager = CacheManager()


class SpecializedCaches:
    """Specialized caches for different data types."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize specialized caches."""
        self.cache_manager = cache_manager
    
    def cache_segmentation_mask(self, image_hash: str, mask: np.ndarray) -> bool:
        """Cache segmentation mask."""
        key = f"segmentation_{image_hash}"
        return self.cache_manager.put(key, mask)
    
    def get_segmentation_mask(self, image_hash: str) -> Optional[np.ndarray]:
        """Get cached segmentation mask."""
        key = f"segmentation_{image_hash}"
        return self.cache_manager.get(key)
    
    def cache_3d_mesh(self, model_hash: str, mesh: trimesh.Trimesh) -> bool:
        """Cache 3D mesh."""
        key = f"mesh_{model_hash}"
        return self.cache_manager.put(key, mesh)
    
    def get_3d_mesh(self, model_hash: str) -> Optional[trimesh.Trimesh]:
        """Get cached 3D mesh."""
        key = f"mesh_{model_hash}"
        return self.cache_manager.get(key)
    
    def cache_layout_params(self, image_hash: str, params: Dict[str, Any]) -> bool:
        """Cache layout parameters."""
        key = f"layout_{image_hash}"
        return self.cache_manager.put(key, params)
    
    def get_layout_params(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached layout parameters."""
        key = f"layout_{image_hash}"
        return self.cache_manager.get(key)
    
    def cache_texture_atlas(self, atlas_hash: str, atlas: np.ndarray) -> bool:
        """Cache texture atlas."""
        key = f"texture_{atlas_hash}"
        return self.cache_manager.put(key, atlas)
    
    def get_texture_atlas(self, atlas_hash: str) -> Optional[np.ndarray]:
        """Get cached texture atlas."""
        key = f"texture_{atlas_hash}"
        return self.cache_manager.get(key)


# Global specialized caches instance
specialized_caches = SpecializedCaches(cache_manager)
