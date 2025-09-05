import asyncio
from collections import deque, defaultdict
from typing import Dict, Any, Deque, Tuple, List, Optional
import time
import logging
from datetime import datetime

# Performance logging
perf_logger = logging.getLogger("perf.frame_broker")


class FrameBroker:
    """In-process pub/sub broker for frames.

    Keeps a small circular buffer per stream and allows subscribers to get
    recent frames via asyncio.Queue.
    """

    def __init__(self, cache_size: int = 16):
        self.cache_size = int(cache_size)
        self._caches: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.cache_size))
        self._subs: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def publish(self, stream_id: str, frame_id: int, ts: float, jpeg: bytes):
        start_time = time.time()
        jpeg_size_kb = len(jpeg) / 1024

        item = {"stream_id": stream_id, "frame_id": int(frame_id), "ts": float(ts), "jpeg": jpeg}
        async with self._locks[stream_id]:
            self._caches[stream_id].append(item)
            # notify subscribers (best-effort)
            subs = list(self._subs.get(stream_id, []))

        subscribers_notified = 0
        for q in subs:
            try:
                q.put_nowait(item)
                subscribers_notified += 1
            except Exception:
                # subscriber queue full or closed — ignore
                pass

        end_time = time.time()
        perf_logger.debug(".3f")

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

    def subscribe(self, stream_id: str, max_queue: int = 8) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=int(max_queue))
        self._subs[stream_id].append(q)
        total_subs = len(self._subs[stream_id])
        perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Subscribed to {stream_id}, queue_size={max_queue}, total_subs={total_subs}")
        return q

    def unsubscribe(self, stream_id: str, q: asyncio.Queue):
        try:
            self._subs[stream_id].remove(q)
            remaining_subs = len(self._subs[stream_id])
            perf_logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Unsubscribed from {stream_id}, remaining_subs={remaining_subs}")
        except Exception:
            perf_logger.warning(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to unsubscribe from {stream_id} - queue not found")


# Global singleton
FRAME_BROKER = FrameBroker()


