"""Guarded image loading for user-provided files."""

from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR):
    """Read an image after checking its file size and decoded dimensions."""
    image_path = Path(path)
    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError("La imagen debe ser un archivo regular, no un enlace simbólico.")
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"La imagen supera el límite de {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("La imagen supera el límite de resolución permitido.")
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError(f"El archivo no es una imagen válida: {image_path.name}") from error
    decoded = cv2.imread(str(image_path), flags)
    if decoded is None:
        raise ValueError(f"OpenCV no pudo leer la imagen: {image_path.name}")
    return decoded
