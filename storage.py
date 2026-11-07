"""Filesystem helpers for the local face dataset."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import unicodedata
from pathlib import Path

from config import DATASET_DIR, LABELS_FILE, MODELS_DIR, RECOGNIZER_MODEL, ROOT_DIR
from image_io import MAX_IMAGE_BYTES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".m4v", ".mpeg", ".mpg", ".webm", ".wmv"}
_ready_cache_key = None
_ready_cache_value = False


def validate_person_name(name: str) -> str:
    """Validate a display name before using it as a directory name."""
    value = unicodedata.normalize("NFC", name.strip())
    if not value or value in {".", ".."}:
        raise ValueError("El nombre no puede estar vacío ni ser una ruta.")
    if len(value) > 80:
        raise ValueError("El nombre no puede superar 80 caracteres.")
    if any(char in value for char in '<>:/\\|?*"'):
        raise ValueError("El nombre no puede contener separadores de ruta.")
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise ValueError("El nombre contiene caracteres no permitidos.")
    if value.endswith((".", " ")):
        raise ValueError("El nombre no puede terminar en punto o espacio.")
    device_name = value.split(".", 1)[0].upper()
    if device_name in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("El nombre está reservado por el sistema operativo.")
    return value


def person_directory(name: str, *, create: bool = False) -> Path:
    """Return a person directory while preventing traversal and symlink escapes."""
    safe_name = validate_person_name(name)
    root = dataset_root()
    target = (root / safe_name).resolve()
    if target.parent != root:
        raise ValueError("La ruta de la persona está fuera del dataset.")
    if target.exists() and not target.is_dir():
        raise ValueError("Ya existe un archivo con ese nombre en el dataset.")
    if create:
        root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            root.parent.chmod(0o700)
            root.chmod(0o700)
            target.chmod(0o700)
    return target


def list_people() -> list[str]:
    root = dataset_root()
    if not root.is_dir():
        return []
    people = []
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            try:
                people.append(validate_person_name(child.name))
            except ValueError:
                continue
    return sorted(people, key=str.casefold)


def dataset_root() -> Path:
    """Resolve the dataset root and reject symlinked project data directories."""
    configured_root = Path(DATASET_DIR)
    project_root = Path(ROOT_DIR).resolve()
    for directory in (project_root / "data", configured_root):
        if directory.is_symlink():
            raise ValueError("Las carpetas de datos no pueden ser enlaces simbólicos.")
    root = configured_root.resolve()
    expected_root = project_root / "data" / "known_faces"
    if root != expected_root:
        raise ValueError("La carpeta de datos debe permanecer en data/known_faces.")
    return root


def list_images(directory: str | Path, *, limit: int | None = None) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        return []
    def candidates():
        for item in path.iterdir():
            if item.suffix.lower() not in IMAGE_EXTENSIONS or item.is_symlink():
                continue
            try:
                if item.is_file() and 0 < item.stat().st_size <= MAX_IMAGE_BYTES:
                    yield item
            except OSError:
                continue

    key = lambda item: item.name.casefold()
    if limit is not None:
        return heapq.nsmallest(max(0, limit), candidates(), key=key)
    return sorted(candidates(), key=key)


def invalidate_recognizer() -> None:
    """Remove generated model files after the dataset changes."""
    global _ready_cache_key, _ready_cache_value
    ensure_models_directory()
    RECOGNIZER_MODEL.unlink(missing_ok=True)
    LABELS_FILE.unlink(missing_ok=True)
    _ready_cache_key = None
    _ready_cache_value = False


def ensure_models_directory(*, create: bool = False) -> Path:
    """Keep generated models inside the project instead of following symlinks."""
    path = Path(MODELS_DIR)
    expected = Path(ROOT_DIR).resolve() / "models"
    if path.is_symlink() or path.resolve() != expected:
        raise ValueError("La carpeta de modelos debe permanecer dentro del proyecto.")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            path.chmod(0o700)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recognizer_ready() -> bool:
    """Check the generated model and label files as a matching pair."""
    global _ready_cache_key, _ready_cache_value
    try:
        ensure_models_directory()
    except (OSError, ValueError):
        return False
    if (RECOGNIZER_MODEL.is_symlink() or LABELS_FILE.is_symlink()
            or not RECOGNIZER_MODEL.is_file() or not LABELS_FILE.is_file()):
        return False
    try:
        model_stat = RECOGNIZER_MODEL.stat()
        labels_stat = LABELS_FILE.stat()
        cache_key = (
            model_stat.st_size,
            model_stat.st_mtime_ns,
            labels_stat.st_size,
            labels_stat.st_mtime_ns,
        )
        if cache_key == _ready_cache_key:
            return _ready_cache_value
        with LABELS_FILE.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict) or "labels" not in metadata:
            _ready_cache_key = cache_key
            _ready_cache_value = False
            return False
        labels = metadata["labels"]
        if sha256_file(RECOGNIZER_MODEL) != metadata.get("model_sha256"):
            _ready_cache_key = cache_key
            _ready_cache_value = False
            return False
        _ready_cache_value = isinstance(labels, dict) and bool(labels)
        _ready_cache_key = cache_key
        return _ready_cache_value
    except (OSError, ValueError, TypeError):
        if "cache_key" in locals():
            _ready_cache_key = cache_key
            _ready_cache_value = False
        return False
