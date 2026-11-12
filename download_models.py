"""
download_models.py
Downloads the required MediaPipe Tasks API .task model files
into the 'models/' directory if they aren't already there.
"""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"[OK] {filename} already present.")
            continue
        print(f"[>>] Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[OK] {filename} saved to {dest}")
        except Exception as e:
            print(f"[✗] Failed to download {filename}: {e}")
            sys.exit(1)
    print("\nAll models ready.\n")


if __name__ == "__main__":
    download_models()
