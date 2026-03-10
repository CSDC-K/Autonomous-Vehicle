"""
Lane Follower — Production (18. MEB Yarışması)

PD kontrol (integral yok — düz pistte gereksiz, sadece gecikme yaratır).
Sabit hız — hız değişimi yok, tüm kontrol direksiyonda.

Pist düzeni:  | || |   (4 çizgi, 2 şerit)
Araç tek şeritte gider; en yakın iki çizgiyi seçer.
"""

import time
import numpy as np


class LaneFollower:
    """
    Mask tabanlı şerit algılama + PD direksiyon kontrolü.
    compute() → (steering, speed, info)
    """

    # ── Yarışma sabitleri ───────────────────────────
    SPEED       = 0.20   # sabit motor gücü (düşük = karar verebilir)
    ALPHA       = 0.85   # konum EMA (1.0=anlık, 0.5=yavaş)
    NO_DET_MAX  = 30     # bu kadar frame tespit yoksa → düz devam
    MIN_LANE_W  = 0.08   # min şerit genişliği (normalize)
    MAX_LANE_W  = 0.65   # max şerit genişliği (normalize)
    MASK_THR    = 0.5    # mask ikili eşik
    MIN_CONF    = 0.85   # %75 altı tespitler hesaba katılmaz
    PROTO_SIZE  = 104    # proto mask çözünürlüğü

    # ── PD gains ────────────────────────────────────
    Kp = 1.50            # orantısal — ana direksiyon gücü
    Kd = 0.50            # türevsel — sallanmayı bastırır, viraj öncesi hazırlar

    def __init__(self, frame_w=416, frame_h=416):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._last_center = 0.5
        self._last_lane = None
        self._no_det = 0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._last_center = 0.5
        self._last_lane = None
        self._no_det = 0
        self._prev_error = 0.0
        self._prev_time = None

    # ── PD hesaplama ────────────────────────────────
    def _pd(self, error):
        now = time.monotonic()
        dt = 0.066 if self._prev_time is None else max(now - self._prev_time, 0.001)
        self._prev_time = now

        p = self.Kp * error
        d = self.Kd * (error - self._prev_error) / dt
        self._prev_error = error

        return max(-1.0, min(1.0, p + d))

    # ── Çizgi pozisyonları ──────────────────────────
    def _extract_positions(self, dets):
        P = self.PROTO_SIZE
        y_start = int(P * 0.35)
        positions = []

        for (box, _conf, cid, mask_104) in dets:
            if _conf < self.MIN_CONF:
                continue
            x1, y1, x2, y2 = box
            mx1 = max(0, int(x1 * P))
            my1 = max(0, int(y1 * P))
            mx2 = min(P, int(x2 * P))
            my2 = min(P, int(y2 * P))

            row0 = max(y_start, (my1 + my2) // 2)
            region = mask_104[row0:my2, mx1:mx2]

            ys, xs = np.where(region > self.MASK_THR)
            if len(xs) >= 3:
                x_norm = (mx1 + xs.mean()) / P
            else:
                x_norm = (x1 + x2) / 2.0

            positions.append((x_norm, int(cid)))

        positions.sort(key=lambda p: p[0])
        return positions

    # ── Şerit seçimi ───────────────────────────────
    def _select_lane(self, positions):
        n = len(positions)
        if n == 0:
            return self._last_lane

        if n == 1:
            x = positions[0][0]
            if self._last_lane is not None:
                old_mid = (self._last_lane[0] + self._last_lane[1]) / 2.0
                old_w = max(0.12, self._last_lane[1] - self._last_lane[0])
                pair = (x, x + old_w) if x < old_mid else (x - old_w, x)
            else:
                pair = (x, x + 0.15) if x < 0.5 else (x - 0.15, x)
            self._last_lane = pair
            return pair

        if n == 2:
            pair = (positions[0][0], positions[1][0])
            self._last_lane = pair
            return pair

        # n >= 3 → en uygun bitişik çifti bul
        best, best_dist = None, float('inf')
        for i in range(n - 1):
            xl, xr = positions[i][0], positions[i + 1][0]
            w = xr - xl
            if w < self.MIN_LANE_W or w > self.MAX_LANE_W:
                continue
            d = abs((xl + xr) / 2.0 - 0.5)
            if d < best_dist:
                best_dist = d
                best = (xl, xr)

        if best is not None:
            self._last_lane = best
            return best

        by_center = sorted(positions, key=lambda p: abs(p[0] - 0.5))
        xs = sorted([by_center[0][0], by_center[1][0]])
        pair = (xs[0], xs[1])
        self._last_lane = pair
        return pair

    # ── Ana hesaplama ───────────────────────────────
    def compute(self, dets):
        positions = self._extract_positions(dets)
        lane = self._select_lane(positions)

        if lane is None:
            self._no_det += 1
            if self._no_det > self.NO_DET_MAX:
                self._prev_error = 0.0
                return 0.0, self.SPEED, {
                    "center": self._last_center, "error": 0,
                    "n_lines": 0, "lane": None
                }
            lane_center = self._last_center
        else:
            self._no_det = 0
            raw = (lane[0] + lane[1]) / 2.0
            lane_center = self.ALPHA * raw + (1.0 - self.ALPHA) * self._last_center
            self._last_center = lane_center

        error = (lane_center - 0.5) * 2.0
        steering = self._pd(error)

        return steering, self.SPEED, {
            "center": lane_center,
            "error": error,
            "steering": steering,
            "speed": self.SPEED,
            "n_lines": len(positions),
            "lane": lane,
            "positions": [p[0] for p in positions],
        }
