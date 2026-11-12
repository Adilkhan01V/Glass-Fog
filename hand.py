"""
hand.py  –  Phase 2: Hand tracking with index-finger-point draw gesture.

Optimized Architecture:
  1. 1-D One Euro filter (Casiez et al., 2012):
     - Dynamic low-pass filter: high jitter suppression when moving slowly,
       virtually zero latency when moving fast.
  2. Result synchronization:
     - Each detection result is processed exactly once; avoids artificial drift
       when display loop runs faster than camera sensor FPS.
  3. Preallocated edge padding:
     - Frame is padded so hands near the borders/corners of the mirror
       are tracked without clipping.
  4. Rotation-invariant reach ratio:
     - Compares fingertip and knuckle distance from wrist in pure float math
       (works whether hand is vertical, tilted, or horizontal).
  5. Debounce + Dropout grace:
     - Bridges short tracking dropouts so fast strokes never break mid-line.

Draw gesture : index finger extended, middle finger curled -> DRAW: ON
Lift gesture : open hand, fist, peace sign, etc.            -> DRAW: OFF
"""

import math
import threading

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


def _cfg(name: str, default):
    """Read tunable from config.py with fallback."""
    return getattr(config, name, default)


# ── Landmark indices ───────────────────────────────────────────────────────────
_WRIST = 0
_INDEX_PIP, _INDEX_TIP = 6, 8
_MIDDLE_PIP, _MIDDLE_TIP = 10, 12


# ── One Euro filter ────────────────────────────────────────────────────────────
def _smoothing_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _OneEuro:
    """1-D One Euro filter (Casiez et al., 2012)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None       # filtered value
        self._raw = None     # previous raw value
        self._dx = 0.0       # filtered velocity

    def __call__(self, x: float, dt: float) -> float:
        if self._x is None:
            self._x = self._raw = x
            return x
        dx = (x - self._raw) / max(dt, 1e-4)
        self._raw = x
        a_d = _smoothing_alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _smoothing_alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


# ── Tracker ────────────────────────────────────────────────────────────────────
class HandTracker:
    """
    Wraps HandLandmarker in LIVE_STREAM mode.

    submit_frame()  — call every frame (non-blocking, fires off async detection)
    get_draw_info() — call every frame to read the latest available result
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = None              # (result, timestamp_ms), written by callback
        self._last_submit_ts = -1
        self._processed_ts = None
        self._padded_buf = None          # reusable zero-allocation border buffer

        # Tunables loaded from config
        self._pad        = float(_cfg("EDGE_PAD", 0.15))
        self._idx_ratio  = float(_cfg("INDEX_EXTENDED_RATIO", 1.15))
        self._mid_ratio  = float(_cfg("MIDDLE_CURLED_RATIO", 1.10))
        self._on_frames  = int(_cfg("GESTURE_ON_FRAMES", 2))
        self._off_frames = int(_cfg("GESTURE_OFF_FRAMES", 2))
        self._grace      = int(_cfg("DROPOUT_GRACE", 3))
        self._min_cutoff = float(_cfg("FILTER_MIN_CUTOFF", 1.2))
        self._beta       = float(_cfg("FILTER_BETA", 0.02))

        self.draw_active = False
        self._clear_tracking()

        conf = float(_cfg("HAND_CONFIDENCE", 0.4))
        base_options = mp_python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
            min_tracking_confidence=conf,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ── Callback (runs on MediaPipe's background thread) ──────────────────────

    def _on_result(self, result, _output_image, timestamp_ms) -> None:
        with self._lock:
            self._latest = (result, timestamp_ms)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pad_px(self, w: int, h: int) -> tuple[int, int]:
        return int(w * self._pad), int(h * self._pad)

    def _clear_tracking(self) -> None:
        self._fx = _OneEuro(self._min_cutoff, self._beta)
        self._fy = _OneEuro(self._min_cutoff, self._beta)
        self._prev_ts = None
        self._on_count = 0
        self._off_count = 0
        self._missed = 0
        self.draw_active = False
        self._out = (False, None, None)

    # ── Public API (call from main thread) ────────────────────────────────────

    def submit_frame(self, frame_rgb: np.ndarray, timestamp_ms: int) -> None:
        """
        Feed an RGB frame. Returns immediately (async detection).
        Guarantees strictly increasing timestamps for MediaPipe.
        Uses reusable preallocated buffer for edge padding.
        """
        if timestamp_ms <= self._last_submit_ts:
            timestamp_ms = self._last_submit_ts + 1
        self._last_submit_ts = timestamp_ms

        h, w = frame_rgb.shape[:2]
        px, py = self._pad_px(w, h)

        if px or py:
            target_shape = (h + 2 * py, w + 2 * px, 3)
            if self._padded_buf is None or self._padded_buf.shape != target_shape:
                self._padded_buf = np.zeros(target_shape, dtype=np.uint8)
            # Copy frame into center region without re-allocating array
            self._padded_buf[py : py + h, px : px + w] = frame_rgb
            input_data = self._padded_buf
        else:
            input_data = frame_rgb

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_data)
        self._landmarker.detect_async(mp_image, int(timestamp_ms))

    def get_draw_info(self, frame_h: int, frame_w: int):
        """
        Returns
        -------
        draw_active : bool
        tip_pt      : (x, y) pixels — filtered index fingertip, or None
        raw_tip_pt  : (x, y) pixels — unsmoothed, for debug, or None
        """
        with self._lock:
            latest = self._latest

        if latest is None:
            return self._out

        result, ts = latest
        if ts == self._processed_ts:
            # Result already processed; return cached state without filter re-churning
            return self._out
        self._processed_ts = ts

        # ── Handle no-hand detection: bridge short dropouts ──────────────────
        if not result.hand_landmarks:
            self._missed += 1
            if self._missed > self._grace:
                self._clear_tracking()
            return self._out
        self._missed = 0

        # ── Landmark coordinates (un-padding back to frame space) ─────────────
        lms = result.hand_landmarks[0]
        px, py = self._pad_px(frame_w, frame_h)
        pw, ph = frame_w + 2 * px, frame_h + 2 * py

        # Extract positions in fast scalar math
        wx  = lms[_WRIST].x * pw - px
        wy  = lms[_WRIST].y * ph - py
        itx = lms[_INDEX_TIP].x * pw - px
        ity = lms[_INDEX_TIP].y * ph - py
        ipx = lms[_INDEX_PIP].x * pw - px
        ipy = lms[_INDEX_PIP].y * ph - py
        mtx = lms[_MIDDLE_TIP].x * pw - px
        mty = lms[_MIDDLE_TIP].y * ph - py
        mpx = lms[_MIDDLE_PIP].x * pw - px
        mpy = lms[_MIDDLE_PIP].y * ph - py

        # Fast Euclidean distances from wrist
        i_tip_dist = math.hypot(itx - wx, ity - wy)
        i_pip_dist = math.hypot(ipx - wx, ipy - wy)
        m_tip_dist = math.hypot(mtx - wx, mty - wy)
        m_pip_dist = math.hypot(mpx - wx, mpy - wy)

        # Reach ratios: index extended, middle curled (rotation invariant!)
        i_reach = i_tip_dist / max(i_pip_dist, 1e-5)
        m_reach = m_tip_dist / max(m_pip_dist, 1e-5)

        gesture = (i_reach > self._idx_ratio) and (m_reach < self._mid_ratio)

        # ── Debounce gesture transitions ─────────────────────────────────────
        if gesture:
            self._on_count += 1
            self._off_count = 0
            if self._on_count >= self._on_frames:
                self.draw_active = True
        else:
            self._off_count += 1
            self._on_count = 0
            if self._off_count >= self._off_frames:
                self.draw_active = False

        # ── Position clamping & One Euro filtering ───────────────────────────
        raw_x = min(max(itx, 0.0), float(frame_w - 1))
        raw_y = min(max(ity, 0.0), float(frame_h - 1))

        dt = 1.0 / 30.0 if self._prev_ts is None else (ts - self._prev_ts) / 1000.0
        dt = min(max(dt, 1e-3), 0.2)
        self._prev_ts = ts

        sx = self._fx(raw_x, dt)
        sy = self._fy(raw_y, dt)

        self._out = (
            self.draw_active,
            (int(sx), int(sy)),
            (int(raw_x), int(raw_y)),
        )
        return self._out

    def reset(self) -> None:
        """Clear tracking history and filters."""
        with self._lock:
            self._latest = None
        self._processed_ts = None
        self._clear_tracking()

    def close(self) -> None:
        self._landmarker.close()
