"""Repository paths, resolved from this file so they work from any cwd."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

MODELS_DIR = REPO_ROOT / "models"
ASSETS_DIR = REPO_ROOT / "assets"

PERSON_MODEL = MODELS_DIR / "yolo11n.pt"
FACE_MODEL = MODELS_DIR / "yolov11n-face.pt"
