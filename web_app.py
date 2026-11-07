"""Local-only browser interface for the face recognition workflow."""

from __future__ import annotations

import argparse
import importlib.util
import io
import logging
import os
import shutil
import sys
import tempfile
import threading
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

from config import (
    FACE_LANDMARKER_MODEL,
    MAX_IMAGES_PER_PERSON,
    POSE_LANDMARKER_MODEL,
    RECOGNIZER_MODEL,
    ROOT_DIR,
)
from detector import (
    boxes_from_landmarks,
    close_detector,
    crop_face,
    detect_faces,
    detect_landmarks,
)
from image_io import MAX_IMAGE_PIXELS
from storage import (
    invalidate_recognizer,
    list_images,
    list_people,
    person_directory,
    recognizer_ready,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
HOST = "127.0.0.1"
PORT = 8765
MAX_REQUEST_BYTES = 5 * 1024 * 1024
ALLOWED_HOSTS = {f"{HOST}:{PORT}", f"localhost:{PORT}"}
ALLOWED_ORIGINS = {f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"}


def load_python_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {path.name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


recognition = load_python_file("face_recognizer_runtime", ROOT_DIR / "4_recognize.py")
training = load_python_file("face_recognizer_training", ROOT_DIR / "3_train.py")

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/assets")
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
logging.getLogger("werkzeug").setLevel(logging.ERROR)

_operation_lock = threading.Lock()
_state_lock = threading.Lock()
_operation: str | None = None
_training_state = {"state": "idle", "message": "", "people": 0, "images": 0}
_loaded_model_stamp: tuple[int, int] | None = None


@app.before_request
def restrict_local_requests():
    if request.host.lower() not in ALLOWED_HOSTS:
        abort(400, description="La aplicación solo acepta conexiones locales.")
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("Origin")
        if origin not in ALLOWED_ORIGINS:
            abort(403, description="Origen de solicitud no permitido.")
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            abort(403, description="No se aceptan solicitudes entre sitios.")


@app.after_request
def add_security_headers(response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data: blob:; connect-src 'self'; media-src 'self' blob:; "
        "base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(409)
@app.errorhandler(413)
@app.errorhandler(415)
@app.errorhandler(422)
def api_error(error):
    return jsonify(error=getattr(error, "description", "La solicitud no pudo completarse.")), error.code


@app.errorhandler(Exception)
def unexpected_error(error):
    if isinstance(error, HTTPException):
        return jsonify(error=error.description), error.code
    app.logger.exception("Error interno en una operación local.")
    return jsonify(error="Ocurrió un error interno. Revisa la terminal local."), 500


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


def _people_summary() -> tuple[list[dict], int]:
    people = list_people()
    result = []
    total_images = 0
    for name in people:
        count = len(list_images(person_directory(name), limit=MAX_IMAGES_PER_PERSON))
        total_images += count
        result.append({"name": name, "images": count})
    return result, total_images


@app.get("/api/status")
def status():
    people, image_total = _people_summary()
    with _state_lock:
        operation = _operation
        training_status = dict(_training_state)
    return jsonify(
        model_ready=recognizer_ready(),
        face_model_ready=FACE_LANDMARKER_MODEL.is_file() and not FACE_LANDMARKER_MODEL.is_symlink(),
        pose_model_ready=POSE_LANDMARKER_MODEL.is_file() and not POSE_LANDMARKER_MODEL.is_symlink(),
        people=len(people),
        images=image_total,
        operation=operation,
        training=training_status,
    )


@app.get("/api/people")
def people_list():
    people, _total = _people_summary()
    return jsonify(people=people)


def _begin_operation(name: str) -> bool:
    global _operation
    if not _operation_lock.acquire(blocking=False):
        return False
    with _state_lock:
        _operation = name
    return True


def _finish_operation() -> None:
    global _operation
    with _state_lock:
        _operation = None
    _operation_lock.release()


def _decode_request_image() -> np.ndarray:
    if request.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        abort(415, description="Envía una imagen JPEG, PNG o WebP.")
    payload = request.get_data(cache=False)
    if not payload:
        abort(400, description="La imagen está vacía.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                abort(413, description="La imagen supera la resolución permitida.")
            source.verify()
        with Image.open(io.BytesIO(payload)) as source:
            rgb = np.asarray(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, ValueError) as error:
        abort(400, description=f"La imagen no es válida: {error}")


def _image_response(frame, *, faces: int):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        abort(500, description="No se pudo preparar la imagen de respuesta.")
    response = app.response_class(encoded.tobytes(), mimetype="image/jpeg")
    response.headers["X-Face-Count"] = str(faces)
    return response


@app.post("/api/people")
def create_person():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400, description="El perfil debe enviarse como un objeto JSON.")
    name = body.get("name")
    if not isinstance(name, str):
        abort(400, description="Escribe un nombre para la persona.")
    if not _begin_operation("profile"):
        abort(409, description="Hay otra operación en curso. Espera un momento.")
    try:
        directory = person_directory(name, create=True)
    except (OSError, ValueError) as error:
        abort(400, description=str(error))
    finally:
        _finish_operation()
    return jsonify(name=directory.name), 201


@app.post("/api/people/<name>/capture")
def capture_person(name: str):
    if not _begin_operation("capture"):
        abort(409, description="Hay otra operación en curso. Espera un momento.")
    try:
        if FACE_LANDMARKER_MODEL.is_symlink() or not FACE_LANDMARKER_MODEL.is_file():
            abort(503, description="Falta el modelo de detección. Ejecuta primero python 1_setup.py.")
        try:
            directory = person_directory(name, create=True)
        except (OSError, ValueError) as error:
            abort(400, description=str(error))
        if not directory.is_dir():
            abort(404, description="No se encontró el perfil seleccionado.")
        images = list_images(directory, limit=MAX_IMAGES_PER_PERSON)
        if len(images) >= MAX_IMAGES_PER_PERSON:
            abort(409, description="Esta persona alcanzó el límite de imágenes.")
        frame = _decode_request_image()
        boxes = detect_faces(frame)
        if not boxes:
            abort(422, description="No se detectó ningún rostro. Ajusta la luz y vuelve a intentar.")
        face = crop_face(frame, max(boxes, key=lambda box: box[2] * box[3]))
        if face.size == 0:
            abort(422, description="No se pudo recortar el rostro.")
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            abort(500, description="No se pudo preparar el recorte facial.")
        invalidate_recognizer()
        filename = f"face_{uuid.uuid4().hex}.jpg"
        destination = directory / filename
        descriptor, temporary_name = tempfile.mkstemp(prefix=".face-", suffix=".tmp", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded.tobytes())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return jsonify(saved=True, images=len(images) + 1), 201
    finally:
        _finish_operation()


@app.delete("/api/people/<name>")
def delete_person(name: str):
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400, description="La confirmación debe enviarse como un objeto JSON.")
    if body.get("confirmation") != "ELIMINAR":
        abort(400, description="Confirma escribiendo ELIMINAR.")
    if not _begin_operation("delete"):
        abort(409, description="Hay otra operación en curso. Espera un momento.")
    try:
        try:
            target = person_directory(name)
        except ValueError as error:
            abort(400, description=str(error))
        if not target.is_dir():
            abort(404, description="No se encontró esa persona.")
        invalidate_recognizer()
        shutil.rmtree(target)
        return jsonify(deleted=True)
    except OSError as error:
        abort(500, description=f"No se pudo eliminar la persona: {error}")
    finally:
        _finish_operation()


@app.post("/api/train")
def start_training():
    global _training_state
    if not _begin_operation("training"):
        abort(409, description="Hay otra operación en curso. Espera a que termine.")
    try:
        people, image_total = _people_summary()
        if not people:
            abort(400, description="Registra al menos una persona antes de entrenar.")
        with _state_lock:
            _training_state = {
                "state": "running",
                "message": "El entrenamiento está en curso.",
                "people": len(people),
                "images": image_total,
            }

        def run_training():
            global _training_state
            try:
                result = training.train()
                with _state_lock:
                    _training_state = {
                        "state": "complete" if result == 0 else "error",
                        "message": "Modelo listo." if result == 0 else "No se pudo completar el entrenamiento.",
                        "people": len(list_people()),
                        "images": sum(
                            len(list_images(person_directory(person), limit=MAX_IMAGES_PER_PERSON))
                            for person in list_people()
                        ),
                    }
            except Exception as error:
                with _state_lock:
                    _training_state = {
                        "state": "error",
                        "message": str(error),
                        "people": 0,
                        "images": 0,
                    }
            finally:
                global _loaded_model_stamp
                _loaded_model_stamp = None
                recognition.recognizer = None
                recognition.label_map = {}
                _finish_operation()

        threading.Thread(target=run_training, name="face-model-training", daemon=True).start()
        return jsonify(state="running", message="El entrenamiento comenzó."), 202
    except Exception:
        _finish_operation()
        raise


@app.post("/api/recognize")
def recognize_image():
    global _loaded_model_stamp
    if not _begin_operation("recognition"):
        abort(409, description="Hay otra operación en curso. Espera un momento.")
    try:
        if not recognizer_ready():
            abort(409, description="Entrena el modelo antes de iniciar el reconocimiento.")
        frame = _decode_request_image()
        stat = RECOGNIZER_MODEL.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if _loaded_model_stamp != stamp or recognition.recognizer is None:
            recognition.load_recognizer()
            _loaded_model_stamp = stamp
        if request.args.get("pose") == "1" and recognition.pose_landmarker is None:
            if POSE_LANDMARKER_MODEL.is_file() and not POSE_LANDMARKER_MODEL.is_symlink():
                recognition.setup_pose(True)
        landmarks = detect_landmarks(frame)
        boxes = boxes_from_landmarks(landmarks, frame.shape)
        result = recognition.process_frame(
            frame,
            mesh=request.args.get("mesh", "1") == "1",
            pose=request.args.get("pose") == "1",
            landmarks=landmarks,
        )
        return _image_response(result, faces=len(boxes))
    finally:
        _finish_operation()


@app.get("/api/output")
def output_info():
    return jsonify(saved=False, message="El análisis web se procesa en memoria y no guarda imágenes.")


def shutdown():
    _operation_lock.acquire()
    try:
        if recognition.pose_landmarker is not None:
            recognition.pose_landmarker.close()
            recognition.pose_landmarker = None
        close_detector()
    finally:
        _operation_lock.release()


def main() -> int:
    global PORT, ALLOWED_HOSTS, ALLOWED_ORIGINS
    parser = argparse.ArgumentParser(description="Inicia la interfaz web local de Face Recognizer.")
    parser.add_argument("--port", type=int, default=PORT, help="Puerto local (predeterminado: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1024 y 65535.")
    PORT = args.port
    ALLOWED_HOSTS = {f"{HOST}:{PORT}", f"localhost:{PORT}"}
    ALLOWED_ORIGINS = {f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"}
    try:
        server = make_server(HOST, PORT, app, threaded=True)
    except OSError as error:
        print(f"No se pudo iniciar el servidor local: {error}", file=sys.stderr)
        return 1
    url = f"http://{HOST}:{PORT}"
    print(f"Face Recognizer disponible en {url}")
    print("El servidor escucha solo en este equipo. Presiona Ctrl+C para salir.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url, new=1)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
