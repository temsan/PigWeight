"""
Simple object tracker using IoU and Hungarian algorithm
Extracted from api/app.py to avoid circular imports

СОЗДАНО: 8 ноября 2025
"""

from typing import Dict, List, Any


class SimpleTracker:
    """
    Простой трекер объектов на основе IoU и венгерского алгоритма
    """
    
    def __init__(self, iou_threshold=0.3, max_age=30, dist_weight=0.2):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.dist_weight = float(dist_weight)
        self.next_id = 1
        # id -> {bbox, age, cx, cy}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _iou(a, b):
        """Вычисляет IoU между двумя bbox"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _center(b):
        """Вычисляет центр bbox"""
        x1, y1, x2, y2 = b
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _hungarian(self, cost: List[List[float]]):
        """Венгерский алгоритм для оптимального сопоставления"""
        n = max(len(cost), len(cost[0]) if cost else 0)
        # pad to square
        C = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                C[i][j] = cost[i][j] if i < len(cost) and j < len(cost[0]) else 1e6
        u = [0.0]*(n+1)
        v = [0.0]*(n+1)
        p = [0]*(n+1)
        way = [0]*(n+1)
        for i in range(1, n+1):
            p[0] = i
            j0 = 0
            minv = [float('inf')]*(n+1)
            used = [False]*(n+1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, n+1):
                    if used[j]:
                        continue
                    cur = C[i0-1][j-1]-u[i0]-v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(0, n+1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        ans = [-1]*n
        for j in range(1, n+1):
            if p[j] != 0 and p[j]-1 < len(cost):
                ans[p[j]-1] = j-1 if j-1 < len(cost[0]) else -1
        return ans

    def update(self, detections: List[Dict[str, Any]]):
        """
        Обновляет треки на основе новых детекций
        
        Args:
            detections: список детекций с bbox
            
        Returns:
            список треков с id
        """
        det_bboxes = [d['bbox'] for d in detections]
        det_centers = [self._center(b) for b in det_bboxes]
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        track_centers = [(self.tracks[tid].get('cx'), self.tracks[tid].get('cy')) for tid in track_ids]

        if det_bboxes and track_bboxes:
            # Build cost = (1 - IoU) + w * normalized center distance
            cost = []
            for i, bb in enumerate(det_bboxes):
                row = []
                cx, cy = det_centers[i]
                for k, tb in enumerate(track_bboxes):
                    iou = self._iou(bb, tb)
                    tcx, tcy = track_centers[k]
                    if tcx is not None and tcy is not None:
                        dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                        # normalize by diagonal
                        diag = (1.0**2 + 1.0**2)**0.5
                        norm_dist = dist / diag if diag > 0 else 0.0
                    else:
                        norm_dist = 0.0
                    c = (1.0 - iou) + self.dist_weight * norm_dist
                    row.append(c)
                cost.append(row)

            assignment = self._hungarian(cost)
            matched_det = set()
            matched_track = set()
            for i, j in enumerate(assignment):
                if j >= 0 and j < len(track_ids):
                    if cost[i][j] < (1.0 - self.iou_threshold):
                        tid = track_ids[j]
                        self.tracks[tid]['bbox'] = det_bboxes[i]
                        self.tracks[tid]['age'] = self.max_age
                        cx, cy = det_centers[i]
                        self.tracks[tid]['cx'] = cx
                        self.tracks[tid]['cy'] = cy
                        matched_det.add(i)
                        matched_track.add(tid)

            # unmatched detections -> new tracks
            for i in range(len(det_bboxes)):
                if i not in matched_det:
                    cx, cy = det_centers[i]
                    self.tracks[self.next_id] = {
                        'bbox': det_bboxes[i],
                        'age': self.max_age,
                        'cx': cx,
                        'cy': cy
                    }
                    self.next_id += 1

            # unmatched tracks -> decrease age
            for tid in track_ids:
                if tid not in matched_track:
                    self.tracks[tid]['age'] -= 2

        else:
            # no existing tracks or no detections
            if det_bboxes:
                for bb in det_bboxes:
                    cx, cy = self._center(bb)
                    self.tracks[self.next_id] = {
                        'bbox': bb,
                        'age': self.max_age,
                        'cx': cx,
                        'cy': cy
                    }
                    self.next_id += 1
            else:
                # no detections, age all
                for tid in track_ids:
                    self.tracks[tid]['age'] -= 2

        # remove old
        to_remove = [tid for tid, t in self.tracks.items() if t['age'] <= 0]
        for tid in to_remove:
            del self.tracks[tid]

        # build output
        out = []
        for tid, t in self.tracks.items():
            x1, y1, x2, y2 = t['bbox']
            out.append({
                'id': tid,
                'bbox': t['bbox'],
                'cx': t.get('cx', 0.5*(x1+x2)),
                'cy': t.get('cy', 0.5*(y1+y2)),
                'age': t['age']
            })
        return out
