"""
main.py  -  Breath Fog Mirror (Optimized & Polished).

Features
--------
- Threaded CameraStream: Decouples camera hardware shutter rate (30 FPS)
  from rendering, allowing a silky smooth, locked 60 FPS output.
- Real-time webcam mirror with 60 FPS cap & non-distorting fullscreen scaling.
- Async LIVE_STREAM detection (FaceLandmarker + HandLandmarker) with zero main-thread stall.
- Mouth "O" detection spawns soft expanding breath condensation and refogs wiped areas.
- Index-finger gesture (index up) draws thin, crisp wiped lines with feathered edges.
- Lift gesture (open hand / fist) moves cursor freely without erasing.
- Natural evaporation: fog slowly evaporates over time like cold window condensation.
- High-res native HUD: Razor-sharp, clean instruction badge in the top-left corner
  with zero blur or muddy shadow, plus toggleable telemetry stats (key 'd').

Controls
--------
Breathe ('O')     - Spawns expanding fog on mirror / refills wiped glass
Point Index       - Wipes/draws thin lines through fog
Lift Hand         - Repositions cursor without wiping
d                 - Toggle debug stats card
f                 - Toggle / fill fog layer manually
c                 - Clear all fog
q / Esc           - Quit
"""

import sys
import time
import os
import threading

import cv2
import numpy as np

import config


def _ensure_models() -> None:
    missing = [p for p in (config.HAND_MODEL_PATH, config.FACE_MODEL_PATH)
               if not os.path.exists(p)]
    if missing:
        print("[!] Model files not found - running download_models.py ...")
        import download_models
        download_models.download_models()


class CameraStream:
    """
    Background camera reader thread.
    Decouples webcam hardware capture speed (many laptop webcams are hardware-capped
    at 30 FPS) from the display & compositing loop, enabling true, fluid 60 FPS.
    """

    def __init__(self, src: int = 0, width: int = config.FRAME_WIDTH, height: int = config.FRAME_HEIGHT) -> None:
        # Use DirectShow on Windows for fastest hardware access
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            print("[X] Cannot open webcam. Check that it is connected and not in use.")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _capture_worker(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=0.5)
        self.cap.release()


# ── Color Palette ─────────────────────────────────────────────────────────────
_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (0,   220, 0)
_CYAN   = (220, 220, 0)
_YELLOW = (0,   220, 220)
_GRAY   = (190, 190, 195)


def _draw_markers(frame: np.ndarray,
                  draw_active: bool, tip_pt,
                  mouth_o_active: bool, mouth_pt) -> None:
    """Draw in-world interactive tracking indicators scaled with the camera frame."""
    # ── Finger Brush Cursor ──────────────────────────────────────────────────
    if tip_pt is not None:
        brush_r = config.WIPE_BRUSH_RADIUS
        if draw_active:
            # Active thin-line eraser ring + center dot
            cv2.circle(frame, tip_pt, brush_r, _GREEN, 2,  cv2.LINE_AA)
            cv2.circle(frame, tip_pt, 2,       _CYAN,  -1, cv2.LINE_AA)
        else:
            # Lifted pen cursor: small indicator circle
            cv2.circle(frame, tip_pt, 6, _WHITE, 1, cv2.LINE_AA)
            cv2.circle(frame, tip_pt, 2, _WHITE, -1, cv2.LINE_AA)

    # ── Mouth "O" Marker (Clean green dot only when open in 'O') ─────────────
    if mouth_pt is not None and mouth_o_active:
        cv2.circle(frame, mouth_pt, 5, _GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, mouth_pt, 5, _BLACK,  1, cv2.LINE_AA)


def _draw_pill_box(img: np.ndarray, x: int, y: int, w: int, h: int,
                   bg_color=(18, 18, 24), border_color=(60, 60, 75), alpha: float = 0.75) -> None:
    """Draws a clean semi-transparent glass card with a subtle border."""
    img_h, img_w = img.shape[:2]
    x2, y2 = min(img_w - 1, x + w), min(img_h - 1, y + h)
    if x2 <= x or y2 <= y:
        return

    roi = img[y:y2, x:x2]
    fill = np.full_like(roi, bg_color)
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)
    cv2.rectangle(img, (x, y), (x2, y2), border_color, 1, cv2.LINE_AA)


def _draw_hud(img: np.ndarray,
              debug_mode: bool,
              fps: float,
              draw_active: bool,
              mouth_o_active: bool,
              scores: dict | None,
              fog_coverage: float) -> None:
    """
    Renders the HUD directly onto the final display canvas at native window resolution.
    Produces razor-sharp, crystal-clear text without blurry bilinear scaling artifacts
    or muddy drop-shadow halos.
    """
    img_h, img_w = img.shape[:2]

    # Dynamically select font scale suited to screen resolution
    scale = 0.48 if img_w < 1200 else 0.54
    font = _FONT

    # ── 1. Clean Instructional Header (Top-Left, Just the Text) ──────────────
    instruction_text = "Breathe 'O' to Fog  |  Point Index to Draw  |  [c] Clear  |  [d] Stats  |  [q] Quit"

    text_x = 16
    text_y = 26

    # Crisp 1px thin shadow for contrast against bright fog, with razor-sharp white text (no black box)
    cv2.putText(img, instruction_text, (text_x + 1, text_y + 1),
                font, scale, (20, 20, 25), 1, cv2.LINE_AA)
    cv2.putText(img, instruction_text, (text_x, text_y),
                font, scale, _WHITE, 1, cv2.LINE_AA)

    # ── 2. Detailed Telemetry Card (Toggleable with 'd') ─────────────────────
    if debug_mode:
        card_x = 16
        card_y = 42
        card_w = 460
        card_h = 136
        line_spacing = 24

        _draw_pill_box(img, card_x, card_y, card_w, card_h,
                       bg_color=(14, 14, 18), border_color=(75, 75, 95), alpha=0.82)

        sy = card_y + 24

        # Performance & Fog
        p_text = f"FPS: {fps:4.1f}  |  Fog: {fog_coverage * 100:3.0f}%  |  Evaporation: {'ON' if config.ENABLE_EVAPORATION else 'OFF'}"
        cv2.putText(img, p_text, (card_x + 14, sy), font, scale * 0.90, _CYAN, 1, cv2.LINE_AA)
        sy += line_spacing

        # Draw Gesture Status
        h_color = _GREEN if draw_active else _YELLOW
        h_text = f"Draw Gesture : {'ACTIVE (erasing)' if draw_active else 'IDLE (pen lifted)'}"
        cv2.putText(img, h_text, (card_x + 14, sy), font, scale * 0.90, h_color, 1, cv2.LINE_AA)
        sy += line_spacing

        # Breath Gesture Status
        m_color = _GREEN if mouth_o_active else _YELLOW
        m_text = f"Mouth 'O'    : {'ACTIVE (fogging)' if mouth_o_active else 'IDLE (mouth resting)'}"
        cv2.putText(img, m_text, (card_x + 14, sy), font, scale * 0.90, m_color, 1, cv2.LINE_AA)
        sy += line_spacing

        # Sensor Blendshapes
        if scores is not None:
            o_val = scores.get("oScore", 0.0)
            b_text = (f"Blendshapes  : jaw={scores['jawOpen']:.2f}  funnel={scores['mouthFunnel']:.2f}  "
                      f"pucker={scores['mouthPucker']:.2f}  score={o_val:.2f}")
            cv2.putText(img, b_text, (card_x + 14, sy), font, scale * 0.85, _GRAY, 1, cv2.LINE_AA)
            sy += line_spacing

        # Resolution & Brush
        r_text = f"Resolution   : {img_w}x{img_h}  |  Brush: {config.WIPE_BRUSH_RADIUS}px  |  Cap: {config.TARGET_FPS} FPS"
        cv2.putText(img, r_text, (card_x + 14, sy), font, scale * 0.85, _WHITE, 1, cv2.LINE_AA)


def main() -> None:
    _ensure_models()

    from hand import HandTracker
    from face import FaceTracker
    from fog import FogRenderer

    camera = CameraStream(src=0, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)

    WIN = "Breath Fog Mirror"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    hand_tracker   = HandTracker()
    face_tracker   = FaceTracker()
    fog_renderer   = FogRenderer(height=config.FRAME_HEIGHT, width=config.FRAME_WIDTH)

    frame_duration = 1.0 / config.TARGET_FPS
    t0_ms          = int(time.perf_counter() * 1000)

    prev_had_hand  = False
    prev_tip_pt    = None
    breath_radius: float | None = None

    debug_mode: bool = config.DEBUG_OVERLAY_DEFAULT
    fps_smooth: float = float(config.TARGET_FPS)
    last_frame_time = time.perf_counter()

    print(f"[OK] Breath Fog Mirror running ({config.TARGET_FPS} FPS Cap, Threaded Camera)")
    print("     - Form an 'O' with your mouth to BREATHE on the mirror")
    print("     - Point index finger to DRAW / WIPE thin lines in the fog")
    print("     - Lift hand (open palm or fist) to NAVIGATE without erasing")
    print("     - Press [d] to toggle stats | [f] Fog | [c] Clear | [q] Quit")

    while True:
        t_start = time.perf_counter()

        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        timestamp_ms = int(time.perf_counter() * 1000) - t0_ms
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── 1. Async Tracking on raw clear frame ─────────────────────────────
        hand_tracker.submit_frame(frame_rgb, timestamp_ms)
        face_tracker.submit_frame(frame_rgb, timestamp_ms)

        draw_active, tip_pt, _ = hand_tracker.get_draw_info(h, w)
        mouth_o_active, mouth_pt, scores = face_tracker.get_mouth_o_info(h, w)

        has_hand = tip_pt is not None
        if prev_had_hand and not has_hand:
            hand_tracker.reset()
            prev_tip_pt = None
        prev_had_hand = has_hand

        fog_renderer.ensure_size(h, w)

        # ── 2. Breath-triggered fog generation ───────────────────────────────
        if mouth_o_active and mouth_pt is not None:
            if breath_radius is None:
                breath_radius = float(config.FOG_BLOB_START_RADIUS)
            else:
                breath_radius = min(
                    float(config.FOG_BLOB_MAX_RADIUS),
                    breath_radius + config.FOG_BLOB_EXPAND_SPEED
                )
            fog_renderer.add_breath_fog(mouth_pt, breath_radius, rate=config.FOG_BLOB_GROWTH_RATE)
        else:
            breath_radius = None

        # ── 3. Finger wipe drawing (thin line strokes) ───────────────────────
        if draw_active and tip_pt is not None:
            fog_renderer.wipe(prev_tip_pt, tip_pt, radius=config.WIPE_BRUSH_RADIUS)
            prev_tip_pt = tip_pt
        else:
            prev_tip_pt = None

        # ── 4. Natural Evaporation ───────────────────────────────────────────
        fog_renderer.evaporate()

        # ── 5. Composite Frosted Fog Layer ───────────────────────────────────
        display_frame = fog_renderer.composite(frame)
        fog_coverage  = float(np.mean(fog_renderer.mask))

        # ── 6. Rolling FPS Calculation ───────────────────────────────────────
        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now
        instant_fps = 1.0 / max(dt, 0.001)
        fps_smooth = 0.92 * fps_smooth + 0.08 * instant_fps

        # ── 7. Draw tracking markers onto in-world frame ─────────────────────
        _draw_markers(display_frame, draw_active, tip_pt, mouth_o_active, mouth_pt)

        # ── 8. Fullscreen-aware display resize ────────────────────────────────
        rect = cv2.getWindowImageRect(WIN)
        win_w, win_h = rect[2], rect[3]
        if win_w > 0 and win_h > 0 and (win_w, win_h) != (w, h):
            display = cv2.resize(display_frame, (win_w, win_h),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            display = display_frame

        # ── 9. Render HUD directly at native window resolution ───────────────
        _draw_hud(display, debug_mode, fps_smooth,
                  draw_active, mouth_o_active, scores, fog_coverage)

        cv2.imshow(WIN, display)

        # ── 10. FPS cap & Key handling ───────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        wait_ms = max(1, int((frame_duration - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            debug_mode = not debug_mode
        elif key == ord("f"):
            fog_renderer.toggle()
        elif key == ord("c"):
            fog_renderer.clear()

    hand_tracker.close()
    face_tracker.close()
    camera.release()
    cv2.destroyAllWindows()
    print("[OK] Exited cleanly.")


if __name__ == "__main__":
    main()
