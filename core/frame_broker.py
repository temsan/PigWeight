import asyncio
from collections import deque, defaultdict
from typing import Dict, Any, Deque, Tuple, List, Optional
import time
import logging
from datetime import datetime
import weakref
import gc

# Performance logging
perf_logger = logging.getLogger("perf.frame_broker")
logger = logging.getLogger(__name__)


class FrameBroker:
    """Enhanced in-process pub/sub broker for frames with backpressure control.

    Features:
    - Circular buffer per stream with configurable size
    - Backpressure control to prevent memory overflow
    - Performance monitoring and metrics
    - Automatic cleanup of stale subscribers
    - Graceful degradation under high load
    """

    def __init__(self, cache_size: int = 16, max_subscribers_per_stream: int = 10):
        self.cache_size = int(cache_size)
        self.max_subscribers_per_stream = int(max_subscribers_per_stream)
        
        # Core data structures
        self._caches: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.cache_size))
        self._subs: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Performance monitoring
        self._publish_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_frames': 0,
            'total_bytes': 0,
            'failed_notifications': 0,
            'successful_notifications': 0,
            'avg_frame_size': 0.0,
            'last_publish_time': 0.0,
            'publish_rate': 0.0
        })
        
        # Backpressure control
        self._backpressure_threshold = 0.8  # Start dropping frames when queues are 80% full
        self._max_queue_size = 32  # Maximum queue size before forced cleanup
        self._cleanup_interval = 60.0  # Cleanup stale subscribers every 60 seconds
        self._last_cleanup = time.time()
        
        # Weak references to track subscriber health
        self._subscriber_refs: Dict[str, List[weakref.ref]] = defaultdict(list)

    async def publish(self, stream_id: str, frame_id: int, ts: float, jpeg: bytes):
        start_time = time.time()
        jpeg_size = len(jpeg)
        jpeg_size_kb = jpeg_size / 1024

        # Periodic cleanup of stale subscribers
        if time.time() - self._last_cleanup > self._cleanup_interval:
            await self._cleanup_stale_subscribers()
            self._last_cleanup = time.time()

        item = {"stream_id": stream_id, "frame_id": int(frame_id), "ts": float(ts), "jpeg": jpeg}
        
        async with self._locks[stream_id]:
            # Add to cache
            self._caches[stream_id].append(item)
            
            # Get current subscribers (copy to avoid modification during iteration)
            subs = list(self._subs.get(stream_id, []))

        # Notify subscribers with backpressure control
        subscribers_notified = 0
        failed_notifications = 0
        
        for q in subs:
            try:
                # Check queue health before publishing
                if self._should_drop_frame(q):
                    failed_notifications += 1
                    continue
                
                q.put_nowait(item)
                subscribers_notified += 1
            except asyncio.QueueFull:
                # Queue is full - apply backpressure
                failed_notifications += 1
                logger.debug(f"Queue full for stream {stream_id}, dropping frame {frame_id}")
            except Exception as e:
                # Subscriber queue closed or other error
                failed_notifications += 1
                logger.debug(f"Failed to notify subscriber for stream {stream_id}: {e}")

        # Update performance statistics
        self._update_publish_stats(stream_id, jpeg_size, subscribers_notified, failed_notifications)

        end_time = time.time()
        publish_duration = end_time - start_time
        
        # Log performance metrics periodically
        if self._publish_stats[stream_id]['total_frames'] % 100 == 0:
            stats = self._publish_stats[stream_id]
            total_notifications = stats['successful_notifications'] + stats['failed_notifications']
            success_rate_str = "100.0%" if total_notifications == 0 else f"{stats['successful_notifications']/total_notifications*100:.1f}%"
            logger.debug(f"Published frame {frame_id} to {len(subs)} subscribers for stream {stream_id}. "
                         f"Queue size: {qsize}, Success rate: {success_rate_str}")

        perf_logger.debug(f"Published frame {frame_id} to {stream_id} in {publish_duration*1000:.1f}ms, "
                         f"size: {jpeg_size_kb:.1f}KB, notified: {subscribers_notified}/{len(subs)}")

    def _should_drop_frame(self, queue: asyncio.Queue) -> bool:
        """Determines if frame should be dropped due to backpressure"""
        try:
            if queue.qsize() >= queue.maxsize * self._backpressure_threshold:
                return True
            return False
        except Exception:
            return True  # If we can't check queue size, assume it's problematic

    async def _get_healthy_subscribers(self, stream_id: str) -> List[asyncio.Queue]:
        """Returns list of healthy subscribers, removing stale ones"""
        healthy_subs = []
        stale_subs = []
        
        for q in self._subs[stream_id]:
            try:
                # Check if queue is still valid
                if hasattr(q, '_closed') and q._closed:  # Queue is closed
                    stale_subs.append(q)
                elif q.qsize() >= self._max_queue_size:  # Queue is overflowing
                    logger.warning(f"Removing overflowing subscriber for stream {stream_id}")
                    stale_subs.append(q)
                else:
                    healthy_subs.append(q)
            except Exception:
                # Queue is in bad state, but be conservative
                healthy_subs.append(q)  # Keep it unless we're sure it's bad
        
        # Remove stale subscribers
        for stale_q in stale_subs:
            try:
                self._subs[stream_id].remove(stale_q)
            except ValueError:
                pass  # Already removed
        
        return healthy_subs

    async def _cleanup_stale_subscribers(self):
        """Periodic cleanup of stale subscribers across all streams"""
        total_removed = 0
        
        for stream_id in list(self._subs.keys()):
            initial_count = len(self._subs[stream_id])
            await self._get_healthy_subscribers(stream_id)  # This removes stale subs
            final_count = len(self._subs[stream_id])
            removed = initial_count - final_count
            total_removed += removed
            
            # Remove empty subscriber lists
            if not self._subs[stream_id]:
                del self._subs[stream_id]
        
        if total_removed > 0:
            logger.info(f"Cleaned up {total_removed} stale subscribers")
            
        # Force garbage collection if we removed many subscribers
        if total_removed > 10:
            gc.collect()

    def _update_publish_stats(self, stream_id: str, frame_size: int, successful: int, failed: int):
        """Updates performance statistics for a stream"""
        stats = self._publish_stats[stream_id]
        current_time = time.time()
        
        stats['total_frames'] += 1
        stats['total_bytes'] += frame_size
        stats['successful_notifications'] += successful
        stats['failed_notifications'] += failed
        
        # Calculate average frame size
        stats['avg_frame_size'] = (stats['total_bytes'] / 1024) / stats['total_frames']
        
        # Calculate publish rate (frames per second)
        if stats['last_publish_time'] > 0:
            time_diff = current_time - stats['last_publish_time']
            if time_diff > 0:
                # Exponential moving average for smooth rate calculation
                alpha = 0.1
                instant_rate = 1.0 / time_diff
                stats['publish_rate'] = alpha * instant_rate + (1 - alpha) * stats['publish_rate']
        
        stats['last_publish_time'] = current_time

    def get_latest(self, stream_id: str) -> Optional[Dict[str, Any]]:
        cache = self._caches.get(stream_id)
        if not cache:
            return None
        try:
            return cache[-1]
        except Exception:
            return None

    def get_all(self, stream_id: str) -> List[Dict[str, Any]]:
        return list(self._caches.get(stream_id, []))
    
    def get_stats(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for streams"""
        if stream_id:
            return {
                'stream_id': stream_id,
                'subscribers': len(self._subs.get(stream_id, [])),
                'cache_size': len(self._caches.get(stream_id, [])),
                'performance': self._publish_stats.get(stream_id, {}).copy()
            }
        else:
            # Return stats for all streams
            return {
                'total_streams': len(self._caches),
                'total_subscribers': sum(len(subs) for subs in self._subs.values()),
                'streams': {
                    sid: {
                        'subscribers': len(self._subs.get(sid, [])),
                        'cache_size': len(cache),
                        'performance': self._publish_stats.get(sid, {}).copy()
                    }
                    for sid, cache in self._caches.items()
                },
                'config': {
                    'cache_size': self.cache_size,
                    'max_subscribers_per_stream': self.max_subscribers_per_stream,
                    'backpressure_threshold': self._backpressure_threshold,
                    'max_queue_size': self._max_queue_size
                }
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the frame broker"""
        total_subs = sum(len(subs) for subs in self._subs.values())
        total_failed = sum(stats.get('failed_notifications', 0) for stats in self._publish_stats.values())
        total_successful = sum(stats.get('successful_notifications', 0) for stats in self._publish_stats.values())
        
        success_rate = 0.0
        if total_successful + total_failed > 0:
            success_rate = total_successful / (total_successful + total_failed)
        
        return {
            'status': 'healthy' if success_rate > 0.9 else 'degraded' if success_rate > 0.7 else 'unhealthy',
            'total_streams': len(self._caches),
            'total_subscribers': total_subs,
            'success_rate': success_rate,
            'total_notifications': total_successful + total_failed,
            'memory_usage_estimate_mb': self._estimate_memory_usage(),
            'last_cleanup': self._last_cleanup
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        total_bytes = 0
        
        # Estimate cache memory
        for cache in self._caches.values():
            for item in cache:
                if 'jpeg' in item:
                    total_bytes += len(item['jpeg'])
                total_bytes += 200  # Estimate for metadata
        
        # Estimate queue memory (rough approximation)
        total_bytes += sum(len(subs) for subs in self._subs.values()) * 1024  # 1KB per queue overhead
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    async def force_cleanup(self):
        """Force cleanup of all stale subscribers and caches"""
        logger.info("Forcing cleanup of FrameBroker")
        await self._cleanup_stale_subscribers()
        
        # Clear empty caches
        empty_streams = [sid for sid, cache in self._caches.items() if not cache]
        for sid in empty_streams:
            del self._caches[sid]
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Cleanup complete. Active streams: {len(self._caches)}, "
                   f"Total subscribers: {sum(len(subs) for subs in self._subs.values())}")

    def subscribe(self, stream_id: str, max_queue: int = 8) -> Optional[asyncio.Queue]:
        # Check subscriber limit
        current_subs = len(self._subs[stream_id])
        if current_subs >= self.max_subscribers_per_stream:
            logger.warning(f"Maximum subscribers ({self.max_subscribers_per_stream}) reached for stream {stream_id}")
            return None
        
        # Create queue with reasonable size limits
        queue_size = min(max(int(max_queue), 4), self._max_queue_size)
        q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        
        self._subs[stream_id].append(q)
        total_subs = len(self._subs[stream_id])
        
        perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Subscribed to {stream_id}, "
                        f"queue_size={queue_size}, total_subs={total_subs}")
        
        return q

    def unsubscribe(self, stream_id: str, q: asyncio.Queue):
        try:
            self._subs[stream_id].remove(q)
            remaining_subs = len(self._subs[stream_id])
            
            # Clean up empty subscriber lists
            if remaining_subs == 0:
                del self._subs[stream_id]
                # Also clean up cache if no subscribers
                if stream_id in self._caches:
                    del self._caches[stream_id]
                # Clean up stats
                if stream_id in self._publish_stats:
                    del self._publish_stats[stream_id]
            
            perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Unsubscribed from {stream_id}, "
                           f"remaining_subs={remaining_subs}")
        except ValueError:
            perf_logger.warning(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to unsubscribe from {stream_id} - queue not found")
        except Exception as e:
            perf_logger.error(f"[{datetime.now().strftime('%H:%M:%S')}] Error unsubscribing from {stream_id}: {e}")


# Global singleton
FRAME_BROKER = FrameBroker()


