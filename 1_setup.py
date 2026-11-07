"""
Paso 1 – Descarga los pesos de YOLOv3-face si no existen.
Ejecuta este script una sola vez antes de empezar.
"""

import os
import tempfile
import shutil
import requests
from tqdm import tqdm

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MAX_RETRIES = 3

FILES = {
    "yolov3-face.cfg":
        "https://raw.githubusercontent.com/sthanhng/yoloface/master/cfg/yolov3-face.cfg",
    "yolov3-face.weights":
        "https://drive.google.com/uc?export=download&id=1xYasjU52whXMLT5MtF7RCPQkV66993oR",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def download(url: str, dest: str):
    """Descarga url → dest con reintentos y escritura atómica via archivo temporal."""
    name = os.path.basename(dest)
    dest_dir = os.path.dirname(os.path.abspath(dest))

    for attempt in range(1, MAX_RETRIES + 1):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dest_dir)
        try:
            print(f"Descargando {name} (intento {attempt}/{MAX_RETRIES}) ...")
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with os.fdopen(tmp_fd, "wb") as f, \
                 tqdm(total=total, unit="B", unit_scale=True) as bar:
                tmp_fd = None  # fdopen takes ownership
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            shutil.move(tmp_path, dest)
            print(f"  Guardado en {dest}\n")
            return
        except Exception as e:
            if tmp_fd is not None:
                os.close(tmp_fd)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            print(f"  Error en intento {attempt}: {e}")
            if attempt == MAX_RETRIES:
                print(f"  Fallaron todos los intentos para {name}. Revisá tu conexión.")
                raise


if __name__ == "__main__":
    for name, url in FILES.items():
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            print(f"  {name} ya existe, omitiendo.")
        else:
            download(url, path)

    print("\nSetup completado.")
    print("Si usas YOLOv3u, el modelo yolov8n.pt se descargara solo la primera vez.")
