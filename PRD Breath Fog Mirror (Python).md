# PRD: Breath Fog Mirror (Python)

*A webcam fun project: breathe on the "glass" to fog it, pinch to draw in the fog.*

## 1. Overview

A real-time Python desktop app that shows the user's webcam feed as a mirror. When the user opens their mouth in an "O" shape, a frosted fog effect appears on the screen, like breathing on a cold window. When the user pinches thumb and index finger together and moves their hand, the fog is wiped away along the path, so they can draw or write in the fog. Breathing again refills the cleared areas with fog.

## 2. Goals and Non-Goals

**Goals**

- Real-time (target 20+ FPS at 640x480) on a normal laptop webcam.
- Realistic, soft frosted-glass look, not flat white overlays.
- Each build phase is runnable and testable on its own.

**Non-Goals**

- No mobile or web version.
- No multi-user or multi-hand support (one face, one hand).
- No saving, sharing or recording features in v1.

## 3. Tech Stack

- Python 3.10 or 3.11, in a virtual environment.
- `opencv-python` for capture, display, blur and compositing.
- `mediapipe` using the **Tasks API** (Face Landmarker with blendshapes enabled, Hand Landmarker). Do not use the legacy `mp.solutions` API.
- `numpy` for the fog mask and pixel math.
- Optional (Phase 7 only): `sounddevice` for blow detection from the microphone.
- Single entry point `main.py`. Keep helpers in small modules (`face.py`, `hand.py`, `fog.py`) once the project grows.
- Provide a `requirements.txt` and short README with run instructions.

## 4. Core Concept: The Fog Mask

Maintain one grayscale float mask the same size as the frame (or half size, upscaled for display). Values range from 0 (clear) to 1 (fully fogged).

- Breathing **adds** to the mask.
- Pinch-drawing **sets values toward 0**.
- Each frame: output = clear frame x (1 - mask) + fog image x mask.
- The fog image is a heavily blurred copy of the current frame, lightly tinted whitish-blue, with subtle noise.

## 5. Functional Requirements

1. **Webcam mirror view.** Show the live webcam feed, flipped horizontally. Flip before running detection so landmark coordinates match the display. Press `q` or `Esc` to quit.
2. **Mouth "O" detection.** Use Face Landmarker blendshapes (`jawOpen` combined with `mouthFunnel` or `mouthPucker`) to decide whether the mouth is in an O shape. Debounce so it must hold for a few consecutive frames before triggering.
3. **Fog spawn.** While the O is held, add a soft, feathered blob to the mask near the mouth (slightly in front of the face). The blob grows the longer the O is held, up to a maximum size.
4. **Pinch detection.** Distance between thumb tip and index tip, normalized by hand size (wrist to middle-finger base). Use hysteresis (a start threshold and a lower end threshold) so the state does not flicker.
5. **Wipe.** While pinched, use the midpoint of the two fingertips as the brush. Draw line segments from the previous to the current position into the mask with a thick, soft-edged brush that sets the mask to 0. Smooth the fingertip position (exponential moving average).
6. **Refog.** Breathing again raises mask values, so wiped areas fill back in.
7. **Evaporation (polish).** The mask decays slightly every frame so fog and wipe lines fade naturally over time.

## 6. Non-Functional Requirements

- Target 20+ FPS at 640x480. Compute the mask and blur at half resolution and upscale if needed.
- Run the hand model with one hand only.
- Handle no-face and no-hand frames gracefully, without crashing.
- Show a small FPS counter and a debug overlay toggle (key `d`) that displays landmark points and the current state (O detected, pinch state).
- Configurable constants (thresholds, brush size, fog growth speed, evaporation rate) kept together at the top of the code or in a config file.

## 7. Build Phases

**Build one phase at a time. Stop after each phase and wait for the user to test it before moving on.** Each phase must run on its own with `python main.py`.

### Phase 1: Project setup and webcam mirror

- Set up venv, `requirements.txt`, and `main.py`.
- Open the webcam, flip horizontally, display in a window, quit on `q`/`Esc`.
- **Test:** window shows a mirrored live feed; moving your right hand appears on the right side of the screen; quitting works cleanly.

### Phase 2: Hand tracking and pinch detection

- Add Hand Landmarker. Draw dots on the thumb tip and index tip, and a larger dot at their midpoint.
- Compute the normalized pinch distance and the pinch state with hysteresis.
- Show "PINCH: ON/OFF" text on screen.
- **Test:** the state flips reliably when you pinch and release, at near and far distances from the camera, with no flicker.

### Phase 3: Face tracking and mouth "O" detection

- Add Face Landmarker with blendshapes. Compute the O-shape state with debounce.
- Show "MOUTH O: ON/OFF" text and the mouth center point on screen.
- **Test:** the state turns ON when you make an O with your mouth, and stays OFF for smiling, talking and a plain open mouth (tune thresholds as needed).

### Phase 4: Static fog layer and compositing

- Build the fog mask and the fog image (blur, tint, noise).
- Temporarily fill the mask fully, with a key (`f`) to toggle fog on and off and `c` to clear it.
- **Test:** pressing `f` shows a convincing frosted-glass look over the live feed; `c` clears it.

### Phase 5: Pinch wipe

- Connect the pinch brush to the mask: pinching and moving clears fog along the path, with soft edges and no gaps during fast motion.
- **Test:** with fog on (`f`), you can draw letters and shapes with your pinch; releasing the pinch stops drawing; lines look smooth, not jittery.

### Phase 6: Breath-triggered fog

- Replace the manual fog key with the mouth O trigger: a growing feathered blob near the mouth while the O is held.
- Wiped areas should refill when you breathe again.
- **Test:** make an O and fog appears and grows; draw in it; breathe again and the drawn lines fill back in.

### Phase 7: Polish and extras

- Add slow evaporation, FPS counter, debug toggle, and tidy config constants.
- Optional realism: faint droplets or streaks near wiped edges.
- Optional bonus: use the microphone (`sounddevice`) to detect an actual blow as an alternative trigger.
- **Test:** runs at target FPS, looks good in normal indoor lighting, and handles missing face or hand without errors.

## 8. Acceptance Criteria (overall)

- Opening the mouth in an O reliably creates fog on screen.
- Pinching thumb and index and moving the hand cleanly wipes fog along the path.
- Breathing again refills wiped areas.
- The app runs smoothly in real time and exits cleanly.
- Each phase can be run and verified independently.

## 9. Risks and Notes

- A hand covering the mouth can make face tracking flicker. Debouncing handles most cases.
- Poor lighting hurts both detectors. Test with the face and hand well lit.
- Mediapipe's legacy solutions API is being replaced. Use the Tasks API, which also provides blendshapes.
- Model files (`.task`) for the Tasks API must be downloaded and referenced by path. Include a small download step or instructions in the README.
