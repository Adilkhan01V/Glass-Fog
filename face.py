"""
face.py  –  Phase 3 (v2): Face tracking and mouth "O" detection, more reliable.

What changed vs v1
  1. Continuous O-score instead of a hard AND of thresholds:
         o_score = min(jawOpen / JAW_THR, max(funnel / FUNNEL_THR, pucker / PUCKER_THR))
     o_score >= 1.0 is exactly the v1 condition, but the score is smoothed and
     fed through hysteresis (separate ON / OFF levels), so a borderline O no
     longer flickers or drops out for a frame.
  2. Each detection result is processed exactly once. v1 debounced and smoothed
     per display frame, so behaviour changed with display FPS.
  3. Short face-dropout grace period: the last state and mouth position are held
     for a few results instead of instantly switching the O off.
  4. One Euro filter on the mouth position (less lag when you move your head).
  5. Optional black border padding (mouth near the frame edge), lower detection
     confidences, strictly increasing timestamps, optional frame skipping.

Public API is unchanged: submit_frame / get_mouth_o_info / reset / close.
get_mouth_o_info() returns scores with one extra key, "oScore" (smoothed), which
is handy for an on-screen debug readout while you tune thresholds.
"""

import math
import threading

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


def _cfg(name, default):
    """Read an optional tunable from config.py, falling back to a default."""
    return getattr(config, name, default)


# Landmark indices for the mouth centre (inner lips)
_UPPER_LIP = 13
_LOWER_LIP = 14


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
        self._x = None
        self._raw = None
        self._dx = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if self._x is None:
            self._x = self._raw = x
            return x
        dx = (x - self._raw) / dt
        self._raw = x
        a_d = _smoothing_alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _smoothing_alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


# ── Tracker ────────────────────────────────────────────────────────────────────
class FaceTracker:
    """
    Wraps FaceLandmarker in LIVE_STREAM mode with blendshapes enabled.

    submit_frame()     — async detection call
    get_mouth_o_info() — debounced O state, mouth centre point, blendshape scores
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = None              # (result, timestamp_ms), set by callback
        self._last_submit_ts = -1
        self._submit_count = 0
        self._processed_ts = None
        self._padded_buf = None

        # Tunables (all optional in config.py except the three v1 thresholds)
        self._pad        = float(_cfg("FACE_EDGE_PAD", 0.10))
        self._every_n    = max(1, int(_cfg("FACE_DETECT_EVERY_N", 1)))
        self._enter      = float(_cfg("MOUTH_O_ENTER", 0.85))   # smoothed score to turn ON
        self._exit       = float(_cfg("MOUTH_O_EXIT", 0.55))    # smoothed score to turn OFF
        self._score_a    = float(_cfg("MOUTH_SCORE_ALPHA", 0.7))
        self._debounce   = max(1, int(config.MOUTH_DEBOUNCE_FRAMES))
        self._grace      = int(_cfg("FACE_DROPOUT_GRACE", 4))
        self._min_cutoff = float(_cfg("FACE_FILTER_MIN_CUTOFF", 1.5))
        self._beta       = float(_cfg("FACE_FILTER_BETA", 0.01))

        self.mouth_o_active: bool = False
        self._clear_tracking()

        conf = float(_cfg("FACE_CONFIDENCE", 0.4))
        base_options = mp_python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=conf,
            min_face_presence_confidence=conf,
            min_tracking_confidence=conf,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # ── Callback (MediaPipe's background thread) ──────────────────────────────

    def _on_result(self, result, _output_image, timestamp_ms) -> None:
        with self._lock:
            self._latest = (result, timestamp_ms)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pad_px(self, w: int, h: int):
        return int(w * self._pad), int(h * self._pad)

    def _clear_tracking(self) -> None:
        self._fx = _OneEuro(self._min_cutoff, self._beta)
        self._fy = _OneEuro(self._min_cutoff, self._beta)
        self._prev_ts = None
        self._score = 0.0
        self._on_count = 0
        self._off_count = 0
        self._missed = 0
        self.mouth_o_active = False
        self._out = (False, None, None)

    # ── Public API ────────────────────────────────────────────────────────────

    def submit_frame(self, frame_rgb: np.ndarray, timestamp_ms: int) -> None:
        """Feed an RGB frame to FaceLandmarker (non-blocking)."""
        self._submit_count += 1
        if (self._submit_count - 1) % self._every_n:
            return                                   # skip this frame (saves CPU)

        if timestamp_ms <= self._last_submit_ts:
            timestamp_ms = self._last_submit_ts + 1
        self._last_submit_ts = timestamp_ms

        h, w = frame_rgb.shape[:2]
        px, py = self._pad_px(w, h)
        if px or py:
            target_shape = (h + 2 * py, w + 2 * px, 3)
            if self._padded_buf is None or self._padded_buf.shape != target_shape:
                self._padded_buf = np.zeros(target_shape, dtype=np.uint8)
            self._padded_buf[py : py + h, px : px + w] = frame_rgb
            input_data = self._padded_buf
        else:
            input_data = frame_rgb
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=input_data
        )
        self._landmarker.detect_async(mp_image, int(timestamp_ms))

    def get_mouth_o_info(self, frame_h: int, frame_w: int):
        """
        Returns
        -------
        mouth_o_active : bool
        mouth_pt       : (x, y) pixels — smoothed mouth centre, or None
        scores         : dict ('jawOpen', 'mouthFunnel', 'mouthPucker', 'oScore') or None
        """
        with self._lock:
            latest = self._latest
        if latest is None:
            return self._out

        result, ts = latest
        if ts == self._processed_ts:
            return self._out                 # nothing new: don't re-process

        self._processed_ts = ts

        # ── No face: hold state briefly, then reset ──────────────────────────
        if not result.face_landmarks or not result.face_blendshapes:
            self._missed += 1
            if self._missed > self._grace:
                self._clear_tracking()
            return self._out
        self._missed = 0

        lms = result.face_landmarks[0]
        scores_all = {b.category_name: b.score for b in result.face_blendshapes[0]}
        jaw_open = scores_all.get("jawOpen", 0.0)
        funnel = scores_all.get("mouthFunnel", 0.0)
        pucker = scores_all.get("mouthPucker", 0.0)

        # ── Continuous O-score (>= 1.0 equals the v1 condition) ──────────────
        jaw_n = jaw_open / max(config.MOUTH_JAW_OPEN_THRESHOLD, 1e-3)
        round_n = max(
            funnel / max(config.MOUTH_FUNNEL_THRESHOLD, 1e-3),
            pucker / max(config.MOUTH_PUCKER_THRESHOLD, 1e-3),
        )
        raw_score = min(max(min(jaw_n, round_n), 0.0), 2.0)
        self._score = self._score_a * raw_score + (1.0 - self._score_a) * self._score

        # ── Hysteresis + debounce (counted per detection result) ─────────────
        if self._score >= self._enter:
            self._on_count += 1
            self._off_count = 0
        elif self._score < self._exit:
            self._off_count += 1
            self._on_count = 0
        # between EXIT and ENTER: keep the current state

        if self._on_count >= self._debounce:
            self.mouth_o_active = True
        if self._off_count >= self._debounce:
            self.mouth_o_active = False

        # ── Mouth centre in frame pixels (undo padding), One Euro smoothed ───
        px, py = self._pad_px(frame_w, frame_h)
        pw, ph = frame_w + 2 * px, frame_h + 2 * py
        raw_mx = (lms[_UPPER_LIP].x + lms[_LOWER_LIP].x) * 0.5 * pw - px
        raw_my = (lms[_UPPER_LIP].y + lms[_LOWER_LIP].y) * 0.5 * ph - py
        raw_mx = min(max(raw_mx, 0.0), frame_w - 1)
        raw_my = min(max(raw_my, 0.0), frame_h - 1)

        dt = 1.0 / 30.0 if self._prev_ts is None else (ts - self._prev_ts) / 1000.0
        dt = min(max(dt, 1e-3), 0.2)
        self._prev_ts = ts

        mouth_pt = (int(self._fx(raw_mx, dt)), int(self._fy(raw_my, dt)))
        scores = {
            "jawOpen": jaw_open,
            "mouthFunnel": funnel,
            "mouthPucker": pucker,
            "oScore": self._score,
        }

        self._out = (self.mouth_o_active, mouth_pt, scores)
        return self._out

    def reset(self) -> None:
        with self._lock:
            self._latest = None
        self._processed_ts = None
        self._clear_tracking()

    def close(self) -> None:
        self._landmarker.close()