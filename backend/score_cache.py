"""
Score cache database for storing evaluated segment scores.
Cleared when new tracks are uploaded.
"""
import hashlib
import json
import os
from datetime import datetime, timedelta

class ScoreCache:
    def __init__(self, cache_file='cache/score_cache.json'):
        self.cache_file = cache_file
        self.cache = {}
        self.load()
    
    def load(self):
        """Load cache from file and clean old entries (>1 day)"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                # Remove entries older than 1 day
                self._cleanup_old_entries()
            except:
                self.cache = {}
    
    def save(self):
        """Save cache to file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_key(self, track1_hash, track2_hash, start1, start2):
        """Generate cache key"""
        return f"{track1_hash}_{track2_hash}_{start1:.1f}_{start2:.1f}"
    
    def get(self, track1_hash, track2_hash, start1, start2):
        """Get cached score"""
        key = self.get_key(track1_hash, track2_hash, start1, start2)
        return self.cache.get(key)
    
    def set(self, track1_hash, track2_hash, start1, start2, score):
        """Store score in cache"""
        key = self.get_key(track1_hash, track2_hash, start1, start2)
        self.cache[key] = {
            'score': score,
            'timestamp': datetime.now().isoformat(),
            'start1': start1,
            'start2': start2
        }
        self.save()
    
    def clear(self):
        """Clear entire cache"""
        self.cache = {}
        self.save()
    
    def clear_by_tracks(self, track1_hash, track2_hash):
        """Clear cache for specific track pair"""
        keys_to_delete = [k for k in self.cache.keys() 
                         if k.startswith(f"{track1_hash}_{track2_hash}_")]
        for key in keys_to_delete:
            del self.cache[key]
        self.save()
    
    def get_all_scores(self, track1_hash, track2_hash):
        """Get all cached scores for a track pair"""
        prefix = f"{track1_hash}_{track2_hash}_"
        return {k: v for k, v in self.cache.items() if k.startswith(prefix)}
    
    def _cleanup_old_entries(self):
        """Remove cache entries older than 1 day"""
        cutoff = datetime.now() - timedelta(days=1)
        keys_to_delete = []
        
        for key, value in self.cache.items():
            try:
                timestamp = datetime.fromisoformat(value.get('timestamp', ''))
                if timestamp < cutoff:
                    keys_to_delete.append(key)
            except (ValueError, TypeError):
                # Invalid timestamp, remove it
                keys_to_delete.append(key)
        
        if keys_to_delete:
            for key in keys_to_delete:
                del self.cache[key]
            self.save()
            print(f"[CACHE] Cleaned up {len(keys_to_delete)} old entries (>1 day)")
    
    def clear_for_new_session(self):
        """Clear cache for new upload session to ensure fresh evaluations"""
        if self.cache:
            print(f"[CACHE] Clearing {len(self.cache)} entries for new session")
            self.clear()

# Global cache instance
_cache = None

def get_cache():
    global _cache
    if _cache is None:
        _cache = ScoreCache()
    return _cache
