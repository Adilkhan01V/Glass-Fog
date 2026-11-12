"""
fog.py  –  Phase 4 (v2): Fog mask management, frosted-glass generation, compositing.

Same look and same public API as v1; only the internals are faster:
  - Compositing uses cv2.blendLinear with 1-channel float weights
    (no 3-channel mask copies, no float multiplies on the full frame).
  - Tint is blended at low resolution (identical result, 16x fewer pixels).
  - The blurred fog layer is cached and refreshed every FOG_REFRESH_EVERY
    frames; it is a heavy blur, so a 1-2 frame old copy is not visible.
  - Evaporation is two in-place OpenCV passes (no temporaries, no np.any scan).
  - Breath-blob falloff kernels are cached per radius instead of rebuilt per frame.
  - Redundant per-frame clips removed (the mask can never exceed FOG_MAX_VALUE:
    only fill() and add_breath_fog() raise it, and both cap it).

Core concept:
  - Mask: 2D float32 array (0.0 = clear, 1.0 = fully frosted).
  - Fog layer: blurred camera feed + cold tint + micro-droplet grain.
"""

import cv2
import numpy as np

import config

# Optional tunable (config.py may define it): rebuild the blurred layer every N frames.
_REFRESH_EVERY = int(getattr(config, "FOG_REFRESH_EVERY", 2))
_DOWNSCALE = 4  # low-res factor for the blur (same as v1)


class FogRenderer:
    """Manages the fog mask and composites the frosted glass effect onto frames."""

    def __init__(self, height: int = config.FRAME_HEIGHT, width: int = config.FRAME_WIDTH) -> None:
        self.height = height
        self.width = width
        self._falloff_cache: dict[int, np.ndarray] = {}
        self._build_buffers()
        self.mask = np.zeros((self.height, self.width), dtype=np.float32)

    # ── Buffers ───────────────────────────────────────────────────────────────

    def _build_buffers(self) -> None:
        h, w = self.height, self.width
        self._small_w = max(16, w // _DOWNSCALE)
        self._small_h = max(16, h // _DOWNSCALE)

        # Weight maps for blendLinear (float32, 1 channel)
        self._inv_mask = np.ones((h, w), dtype=np.float32)

        # Tint at low-res: blending a constant colour commutes with the upscale,
        # so this matches v1 while touching 16x fewer pixels.
        self._tint_small = np.full(
            (self._small_h, self._small_w, 3), config.FOG_TINT_COLOR, dtype=np.uint8
        )

        # Micro-droplet grain (uint8), generated once
        rng = np.random.RandomState(42)
        noise_range = max(1, int(config.FOG_NOISE_INTENSITY) * 2)
        self._noise_u8 = rng.randint(0, noise_range, (h, w, 3), dtype=np.uint8)

        # Output + cached fog layer
        self._out = np.empty((h, w, 3), dtype=np.uint8)
        self._fog_cache = None
        self._fog_age = 10**9

        self._ksize = max(1, (config.FOG_BLUR_SIZE // _DOWNSCALE) * 2 + 1)

    def ensure_size(self, height: int, width: int) -> None:
        """Resize mask and internal buffers if the frame dimensions change."""
        if self.height != height or self.width != width:
            old_mask = self.mask
            self.height = height
            self.width = width
            self._build_buffers()
            self.mask = cv2.resize(old_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    # ── Mask operations ───────────────────────────────────────────────────────

    def fill(self, val: float = 1.0) -> None:
        """Fill the entire mirror with fog."""
        self.mask.fill(min(val, config.FOG_MAX_VALUE))

    def clear(self) -> None:
        """Clear all fog completely."""
        self.mask.fill(0.0)

    def evaporate(self, rate: float = config.FOG_EVAPORATION_RATE) -> None:
        """Dissipate fog over time: mask = max(mask - rate, 0), in place."""
        if not config.ENABLE_EVAPORATION:
            return
        cv2.subtract(self.mask, rate, dst=self.mask)
        # values <= 0 become 0 (cheap in-place clamp, no temporaries)
        cv2.threshold(self.mask, 0.0, 0.0, cv2.THRESH_TOZERO, dst=self.mask)

    def _falloff(self, r: int) -> np.ndarray:
        """Cached radial kernel (1 - d^2/R^2)^2 of size (2r+1, 2r+1)."""
        k = self._falloff_cache.get(r)
        if k is None:
            ax = np.arange(-r, r + 1, dtype=np.float32)
            dist_sq = ax[None, :] ** 2 + ax[:, None] ** 2
            k = (1.0 - np.clip(dist_sq / float(r * r), 0.0, 1.0)) ** 2
            k = k.astype(np.float32)
            if len(self._falloff_cache) > 256:
                self._falloff_cache.clear()
            self._falloff_cache[r] = k
        return k

    def add_breath_fog(self, center: tuple[int, int], radius: float, rate: float = config.FOG_BLOB_GROWTH_RATE) -> None:
        """
        Add a soft, feathered breath condensation blob around the mouth center.
        Builds density frame-by-frame while breathing; refills wiped lines.
        """
        cx, cy = center
        r = int(radius)
        if r <= 0:
            return

        x0 = max(0, cx - r)
        x1 = min(self.width, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(self.height, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return

        kernel = self._falloff(r)
        kx0, ky0 = x0 - (cx - r), y0 - (cy - r)
        kernel = kernel[ky0 : ky0 + (y1 - y0), kx0 : kx0 + (x1 - x0)]

        roi = self.mask[y0:y1, x0:x1]
        roi += kernel * rate
        np.minimum(roi, config.FOG_MAX_VALUE, out=roi)

    def wipe(self, pt1: tuple[int, int] | None, pt2: tuple[int, int], radius: int = config.WIPE_BRUSH_RADIUS) -> None:
        """
        Wipe away fog between pt1 and pt2 with soft feathered edges.
        If pt1 is None, wipes a single dot at pt2. Uses np.minimum on the local
        ROI so existing wipe marks are never re-fogged by overlapping strokes.
        """
        p1 = pt1 if pt1 is not None else pt2
        p2 = pt2

        margin = int(1.4 * radius) + 4
        x_min = max(0, min(p1[0], p2[0]) - margin)
        x_max = min(self.width, max(p1[0], p2[0]) + margin)
        y_min = max(0, min(p1[1], p2[1]) - margin)
        y_max = min(self.height, max(p1[1], p2[1]) + margin)
        if x_max <= x_min or y_max <= y_min:
            return

        roi_w = x_max - x_min
        roi_h = y_max - y_min
        local_p1 = (p1[0] - x_min, p1[1] - y_min)
        local_p2 = (p2[0] - x_min, p2[1] - y_min)

        stroke_roi = np.ones((roi_h, roi_w), dtype=np.float32)

        # Outer soft halo -> mid transition -> solid clear core
        for r_factor, val in ((1.25, 0.40), (0.95, 0.15), (0.65, 0.00)):
            curr_r = max(1, int(radius * r_factor))
            cv2.line(stroke_roi, local_p1, local_p2, float(val), thickness=2 * curr_r, lineType=cv2.LINE_AA)
            cv2.circle(stroke_roi, local_p1, curr_r, float(val), -1, lineType=cv2.LINE_AA)
            cv2.circle(stroke_roi, local_p2, curr_r, float(val), -1, lineType=cv2.LINE_AA)

        roi = self.mask[y_min:y_max, x_min:x_max]
        np.minimum(roi, stroke_roi, out=roi)

    def toggle(self) -> None:
        """Toggle between full fog and clear."""
        if float(np.mean(self.mask)) > 0.5:
            self.clear()
        else:
            self.fill(1.0)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def create_fog_layer(self, frame: np.ndarray) -> np.ndarray:
        """
        Frosted-glass layer from the current frame:
          1. Downscale + box blur at low-res
          2. Cold tint blended at low-res
          3. Upscale, then add micro-droplet grain
        """
        small = cv2.resize(frame, (self._small_w, self._small_h), interpolation=cv2.INTER_NEAREST)
        blurred_small = cv2.blur(small, (self._ksize, self._ksize))

        w_tint = config.FOG_TINT_WEIGHT
        tinted_small = cv2.addWeighted(blurred_small, 1.0 - w_tint, self._tint_small, w_tint, 0)

        fog_layer = cv2.resize(tinted_small, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return cv2.add(fog_layer, self._noise_u8, dst=fog_layer)

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the camera frame with the frosted layer using self.mask.
        Returns an internal buffer that is overwritten on the next call.
        """
        # Fast exit if there is no fog at all (single cheap pass)
        if cv2.minMaxLoc(self.mask)[1] <= 0.005:
            return frame

        if self._fog_cache is None or self._fog_age >= _REFRESH_EVERY:
            self._fog_cache = self.create_fog_layer(frame)
            self._fog_age = 0
        self._fog_age += 1

        np.subtract(1.0, self.mask, out=self._inv_mask)
        return cv2.blendLinear(frame, self._fog_cache, self._inv_mask, self.mask, self._out)