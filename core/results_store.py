import os
import time
import logging
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List
from datetime import datetime

# Performance logging
perf_logger = logging.getLogger("perf.results_store")


class ResultsStore:
    """In-memory results store with TTL for inference outputs.

    Stores results per stream_id keyed by frame_id. Keeps small history with TTL.
    """

    def __init__(self, ttl_s: float = None, history_len: int = None):
        # allow overriding from environment
        try:
            ttl_env = os.getenv('RESULTS_TTL')
            ttl_s = float(ttl_env) if ttl_env is not None else (ttl_s if ttl_s is not None else 5.0)
        except Exception:
            ttl_s = ttl_s if ttl_s is not None else 5.0
        try:
            hist_env = os.getenv('RESULTS_HISTORY')
            history_len = int(hist_env) if hist_env is not None else (history_len if history_len is not None else 32)
        except Exception:
            history_len = history_len if history_len is not None else 32

        self.ttl_s = float(ttl_s)
        self.history_len = int(history_len)
        # stream_id -> deque of (frame_id, ts, result)
        self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_len))

    def put(self, stream_id: str, frame_id: int, result: Dict[str, Any]):
        start_time = time.time()
        detections = result.get('detections', 0)
        confidence = result.get('confidence', 0.0)

        now = time.time()
        self._store[stream_id].append((int(frame_id), now, result))

        current_count = len(self._store[stream_id])
        perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Stored result for {stream_id} frame {frame_id}, detections={detections}, confidence={confidence:.2f}, store_size={current_count}")

    def get_latest(self, stream_id: str) -> Optional[Dict[str, Any]]:
        items = self._store.get(stream_id)
        if not items:
            perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] No results for {stream_id}")
            return None

        # purge expired
        cutoff = time.time() - self.ttl_s
        expired_count = 0
        while items and items[0][1] < cutoff:
            items.popleft()
            expired_count += 1

        if expired_count > 0:
            perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Cleaned {expired_count} expired results for {stream_id}")

        if not items:
            perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] All results expired for {stream_id}")
            return None

        latest_result = items[-1][2]
        detections = latest_result.get('detections', 0)
        perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Retrieved latest result for {stream_id}, detections={detections}, remaining={len(items)}")
        return latest_result

    def get_for_frame(self, stream_id: str, frame_id: int) -> Optional[Dict[str, Any]]:
        items = self._store.get(stream_id)
        if not items:
            perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] No results for {stream_id}")
            return None

        cutoff = time.time() - self.ttl_s
        for fid, ts, res in reversed(items):
            if ts < cutoff:
                break
            if fid == int(frame_id):
                detections = res.get('detections', 0)
                perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Retrieved result for {stream_id} frame {frame_id}, detections={detections}")
                return res

        perf_logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] No result found for {stream_id} frame {frame_id}")
        return None


# Singleton
RESULTS_STORE = ResultsStore()


