"""
config.py
All tuneable constants for Breath Fog Mirror.
"""

# ── Display ───────────────────────────────────────────────────────────────────
TARGET_FPS: int = 60          # hard cap
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

# ── Fog mask ──────────────────────────────────────────────────────────────────
ENABLE_EVAPORATION: bool = True       # naturally evaporate fog over time
FOG_EVAPORATION_RATE: float = 0.0004   # fraction subtracted per frame (~40s full cycle at 60fps)
FOG_MAX_VALUE: float = 1.0

# ── Debug & UI ────────────────────────────────────────────────────────────────
DEBUG_OVERLAY_DEFAULT: bool = False   # press 'd' to toggle on-screen debug stats

# ── Fog Appearance & Compositing ──────────────────────────────────────────────
FOG_BLUR_SIZE: int = 41               # blur kernel size for frosted view
FOG_TINT_COLOR = (245, 235, 220)       # BGR: subtle cold whitish-blue tint
FOG_TINT_WEIGHT: float = 0.30         # strength of tint (0 = pure blur, 1 = solid color)
FOG_NOISE_INTENSITY: float = 8.0      # frosted glass micro-droplet grain (0 = smooth)

# ── Breath / Mouth-O detection ────────────────────────────────────────────────
MOUTH_JAW_OPEN_THRESHOLD: float = 0.35
MOUTH_FUNNEL_THRESHOLD:   float = 0.25
MOUTH_PUCKER_THRESHOLD:   float = 0.25
MOUTH_DEBOUNCE_FRAMES:    int = 3         # frames needed to trigger ON / OFF
MOUTH_O_ENTER:            float = 0.85    # smoothed O-score to turn ON
MOUTH_O_EXIT:             float = 0.55    # smoothed O-score to turn OFF
MOUTH_SCORE_ALPHA:        float = 0.70    # smoothing factor for continuous O-score
FACE_EDGE_PAD:            float = 0.10    # padding fraction for edge detection
FACE_DETECT_EVERY_N:      int = 1         # run face detection every N frames
FACE_DROPOUT_GRACE:       int = 4         # missed frames to bridge
FACE_FILTER_MIN_CUTOFF:   float = 1.5     # One Euro min cutoff for mouth position
FACE_FILTER_BETA:         float = 0.01    # One Euro speed factor
FACE_CONFIDENCE:          float = 0.40    # min detection/tracking confidence

# ── Fog spawn (breath) ────────────────────────────────────────────────────────
FOG_BLOB_GROWTH_RATE: float = 0.06    # fog density added to mask per frame while O held
FOG_BLOB_START_RADIUS: int = 50       # initial breath radius in pixels
FOG_BLOB_MAX_RADIUS: int = 190        # maximum breath spread radius in pixels
FOG_BLOB_EXPAND_SPEED: float = 2.5    # radius growth per frame while breathing
FOG_REFRESH_EVERY: int = 2            # recompute heavy blur layer every N frames (caching)

# ── Wipe brush ────────────────────────────────────────────────────────────────
WIPE_BRUSH_RADIUS: int = 9            # fine fingertip brush for thin lines
WIPE_BRUSH_SOFTNESS: float = 0.6      # 0 = hard edge, 1 = fully Gaussian
FINGER_SMOOTH_ALPHA: float = 0.72     # EMA weight for fingertip position

# ── Hand Tracking & Gesture Filtering (One Euro Filter) ───────────────────────
EDGE_PAD: float = 0.15                # padding fraction for robust edge/corner tracking
INDEX_EXTENDED_RATIO: float = 1.15    # index tip/pip reach ratio to enter drawing
MIDDLE_CURLED_RATIO: float = 1.10     # middle finger curled reach threshold
GESTURE_ON_FRAMES: int = 2            # debounce frames to start drawing
GESTURE_OFF_FRAMES: int = 2           # debounce frames to lift pen
DROPOUT_GRACE: int = 3                # frames to bridge brief tracking dropouts
FILTER_MIN_CUTOFF: float = 1.2        # One Euro filter min cutoff (low-jitter when slow)
FILTER_BETA: float = 0.02             # One Euro filter speed coefficient (zero-lag when fast)

# ── Model paths ───────────────────────────────────────────────────────────────
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
FACE_MODEL_PATH: str = _os.path.join(_HERE, "models", "face_landmarker.task")
HAND_MODEL_PATH: str = _os.path.join(_HERE, "models", "hand_landmarker.task")
