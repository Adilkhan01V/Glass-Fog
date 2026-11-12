# Breath Fog Mirror

A real-time desktop app that turns your webcam into a cold window: **breathe** on the glass to fog it up, **point your finger** to draw or write in the fog, and watch it **naturally evaporate** over time!

---

## Features

- **Live Mirror Feed**: Horizontally mirrored, rendered at up to **60 FPS**, and resized to the current window.
- **Breath-Triggered Fog**: Form an "O" shape with your mouth to exhale warm breath. A soft-feathered condensation cloud spawns and expands outward.
- **Finger Wipe Drawing**: Point your index finger to erase fog with a fine brush stroke and soft condensation edges.
- **Pen-Lift Navigation**: Open your hand or make a fist to reposition your hand freely without wiping.
- **Refogging**: Breathe again over wiped lines or drawings to refill them with fresh condensation.
- **Natural Evaporation**: Fog gently dissipates over 30–40 seconds, just like physical condensation in a warm room.
- **Asynchronous Tracking**: Uses the MediaPipe Tasks API (`LIVE_STREAM` mode) to keep tracking work from blocking the display loop.
- **Minimal View & Debug HUD**: Clean display by default; press `d` to toggle the telemetry overlay.

---

## Requirements

- Python **3.10 – 3.12**
- Built-in or USB webcam
- Windows, macOS, or Linux desktop

---

## Run on Windows

Open PowerShell in the project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, allow it for the current terminal only, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The model files are included in `models/`. If either model file is missing, the app attempts to download it on startup. A webcam must be connected and available to the app.

---

## Gestures & Controls

| Action | How to Trigger |
|--------|----------------|
| **Breathe (Fog Glass)** | Open mouth in an "O" shape |
| **Draw / Erase** | Extend your index finger and curl your middle finger |
| **Lift Pen / Move Freely** | Open your hand flat, make a fist, or peace sign |
| **Toggle Debug HUD** | Press `d` |
| **Toggle Fog On/Off** | Press `f` |
| **Clear All Fog** | Press `c` |
| **Quit** | Press `q` or `Esc` |

---

## Project Structure

```
Fog Project/
├── main.py               # Main application loop & fullscreen display
├── config.py             # All tuneable constants & thresholds
├── fog.py                # Fog mask, realistic frosted glass & compositing
├── face.py               # FaceLandmarker with blendshapes & mouth-O detection
├── hand.py               # HandLandmarker with index-finger drawing gesture
├── download_models.py    # Auto-downloads MediaPipe .task models
├── requirements.txt      # Project dependencies
└── models/               # MediaPipe Tasks model files
    ├── face_landmarker.task
    └── hand_landmarker.task
```

---

## Configuration (`config.py`)

All tuneable constants are cleanly organized in [`config.py`](config.py):

- `TARGET_FPS = 60` — Frame rate limiter
- `WIPE_BRUSH_RADIUS = 9` — Eraser tip size in pixels
- `FOG_BLOB_GROWTH_RATE = 0.06` — Breath density accumulation speed
- `FOG_BLOB_EXPAND_SPEED = 2.5` — Breath radius expansion speed
- `ENABLE_EVAPORATION = True` — Natural condensation evaporation toggle
- `FOG_EVAPORATION_RATE = 0.0004` — Evaporation decay per frame
- `DEBUG_OVERLAY_DEFAULT = False` — Initial state of debug HUD
