"""
Paso 4 – Reconocimiento facial con MediaPipe Tasks API (0.10.x).
Malla facial (478 puntos) + esqueleto corporal + LBPH.

Uso:
  python 4_recognize.py                    # webcam
  python 4_recognize.py --source foto.jpg  # imagen
  python 4_recognize.py --source video.mp4 # video
  python 4_recognize.py --source carpeta/  # carpeta
  python 4_recognize.py --no-mesh --no-pose
"""

import os, json, argparse, time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from detector import detect_faces, crop_face
from config import RECOGNIZER_MODEL, LABELS_FILE, RECOGNIZE_THRESHOLD, FACE_SIZE

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--source",   default="webcam")
parser.add_argument("--save",     action="store_true")
parser.add_argument("--no-pose",  action="store_true")
parser.add_argument("--no-mesh",  action="store_true")
args = parser.parse_args()

# ─── LBPH ─────────────────────────────────────────────────────────────────────
recognizer = None
label_map  = {}

if os.path.exists(RECOGNIZER_MODEL):
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(RECOGNIZER_MODEL)
        if not os.path.exists(LABELS_FILE):
            raise FileNotFoundError(f"Archivo de etiquetas no encontrado: '{LABELS_FILE}'")
        with open(LABELS_FILE) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        print(f"Personas: {list(label_map.values())}")
    except Exception as e:
        print(f"⚠  Error al cargar el modelo: {e}")
        print("   Re-entrenando puede resolver el problema. Solo análisis visual por ahora.")
        recognizer = None
        label_map = {}
else:
    print("Modelo no entrenado — solo análisis visual.")

print(f"Umbral: {RECOGNIZE_THRESHOLD}  |  Q = salir\n")

# ─── MediaPipe Tasks ───────────────────────────────────────────────────────────
FACE_MODEL = "models/face_landmarker.task"
POSE_MODEL = "models/pose_landmarker.task"

face_landmarker = None
pose_landmarker = None

if not args.no_mesh and os.path.exists(FACE_MODEL):
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)
elif not args.no_mesh:
    print("⚠  face_landmarker.task no encontrado. Ejecutá 1_setup.py")

if not args.no_pose and os.path.exists(POSE_MODEL):
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)
elif not args.no_pose:
    print("⚠  pose_landmarker.task no encontrado. Ejecutá 1_setup.py")

# ─── Conexiones de pose (33 landmarks COCO) ───────────────────────────────────
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
]

# Conexiones faciales clave (contornos)
# Índices de los 468 landmarks: ojos, cejas, nariz, labios, óvalo
FACE_OVAL   = [10,338,297,332,284,251,389,356,454,323,361,288,
               397,365,379,378,400,377,152,148,176,149,150,136,
               172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE    = [362,382,381,380,374,373,390,249,263,466,388,387,
               386,385,384,398]
RIGHT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,
               159,160,161,246]
LEFT_BROW   = [276,283,282,295,285,300,293,334,296,336]
RIGHT_BROW  = [46,53,52,65,55,70,63,105,66,107]
NOSE        = [1,2,98,327]
LIPS_OUTER  = [61,146,91,181,84,17,314,405,321,375,291,
               308,324,318,402,317,14,87,178,88,95,78]

def _connect(frame, lms, indices, color, thick=1):
    h, w = frame.shape[:2]
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], color, thick, cv2.LINE_AA)

def _dot(frame, lms, indices, color, r=1):
    h, w = frame.shape[:2]
    for i in indices:
        x, y = int(lms[i].x * w), int(lms[i].y * h)
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)

# ─── Reconocimiento LBPH ──────────────────────────────────────────────────────
def recognize(face_bgr):
    if recognizer is None:
        return "Desconocido", 0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE)
    gray = cv2.equalizeHist(gray)
    label_id, dist = recognizer.predict(gray)
    conf = max(0, (1 - dist / RECOGNIZE_THRESHOLD) * 100)
    name = label_map.get(label_id, "?") if dist < RECOGNIZE_THRESHOLD else "Desconocido"
    return name, conf

def draw_label(frame, x, y, w, h, name, conf):
    known = name != "Desconocido"
    color = (0, 220, 80) if known else (60, 60, 200)

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Esquinas
    L = max(12, w // 6)
    for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame, (px, py), (px+dx*L, py), color, 3)
        cv2.line(frame, (px, py), (px, py+dy*L), color, 3)

    label = f"{name}  {conf:.0f}%" if known else name
    fs = 0.55
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.rectangle(frame, (x, y-th-10), (x+tw+8, y), color, -1)
    cv2.putText(frame, label, (x+4, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

# ─── Pipeline ─────────────────────────────────────────────────────────────────
def process_frame(frame, boxes=None):
    """
    boxes: lista de (x,y,w,h) pre-calculada; si es None se detecta en este frame.
    Pasarla desde process_video evita re-detectar en frames intermedios.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ── Pose corporal ─────────────────────────────────────────────────────────
    if pose_landmarker:
        pose_res = pose_landmarker.detect(mp_img)
        for person in pose_res.pose_landmarks:
            h, w = frame.shape[:2]
            # Conexiones
            for a, b in POSE_CONNECTIONS:
                if a < len(person) and b < len(person):
                    x1 = int(person[a].x * w); y1 = int(person[a].y * h)
                    x2 = int(person[b].x * w); y2 = int(person[b].y * h)
                    cv2.line(frame, (x1,y1), (x2,y2), (50, 200, 255), 2, cv2.LINE_AA)
            # Puntos
            for lm in person:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 80, 80), -1, cv2.LINE_AA)

    # ── Malla facial ──────────────────────────────────────────────────────────
    if face_landmarker:
        face_res = face_landmarker.detect(mp_img)
        for face_lms in face_res.face_landmarks:
            h, w = frame.shape[:2]

            # Todos los puntos como malla tenue
            for lm in face_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 100, 160), -1)

            # Contornos clave con colores
            _connect(frame, face_lms, FACE_OVAL,  (0, 180, 255), 1)
            _connect(frame, face_lms, LEFT_EYE,   (0, 255, 180), 1)
            _connect(frame, face_lms, RIGHT_EYE,  (0, 255, 180), 1)
            _connect(frame, face_lms, LEFT_BROW,  (180, 255, 0), 1)
            _connect(frame, face_lms, RIGHT_BROW, (180, 255, 0), 1)
            _connect(frame, face_lms, LIPS_OUTER, (255, 80, 180), 1)
            _dot(frame, face_lms, NOSE, (255, 255, 0), 2)

            # Iris (landmarks 468-477)
            if len(face_lms) >= 478:
                for iris_idx in range(468, 478):
                    cx = int(face_lms[iris_idx].x * w)
                    cy = int(face_lms[iris_idx].y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1, cv2.LINE_AA)

    # ── Reconocimiento LBPH ───────────────────────────────────────────────────
    if boxes is None:
        boxes = detect_faces(frame)
    for box in boxes:
        face_crop = crop_face(frame, box)
        if face_crop.size == 0:
            continue
        name, conf = recognize(face_crop)
        draw_label(frame, *box, name, conf)

    return frame

# ─── Modos de entrada ─────────────────────────────────────────────────────────
def process_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"No se pudo leer: {path}"); return
    result = process_frame(frame)
    if args.save:
        os.makedirs("output", exist_ok=True)
        out = os.path.join("output", os.path.basename(path))
        cv2.imwrite(out, result)
        print(f"Guardado: {out}")
    cv2.imshow("Face Recognizer", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_folder(folder):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    imgs = sorted([os.path.join(folder, f)
                   for f in os.listdir(folder) if f.lower().endswith(exts)])
    for path in imgs:
        process_image(path)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def process_video(source):
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir '{source}'"); return

    out_writer = None
    fname = None
    if args.save:
        os.makedirs("output", exist_ok=True)
        fname  = "output/webcam.avi" if source == "webcam" else \
                 os.path.join("output", os.path.basename(source))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 20
        W, H   = int(cap.get(3)), int(cap.get(4))
        out_writer = cv2.VideoWriter(fname, fourcc, fps, (W, H))

    frame_skip, frame_idx = 3, 0
    boxes = []
    fps_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Detección solo cada frame_skip frames; los intermedios reusan boxes
            if frame_idx % frame_skip == 0:
                boxes = detect_faces(frame)

            result = process_frame(frame.copy(), boxes=boxes)

            # FPS en esquina superior derecha
            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - fps_time, 1e-6))
            fps_time = now
            cv2.putText(result, f"{fps_display:.1f} fps",
                        (result.shape[1] - 100, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            if out_writer:
                out_writer.write(result)

            cv2.imshow("Face Recognizer  (Q = salir)", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
            print(f"Guardado en {fname}")
        cv2.destroyAllWindows()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    src = args.source
    if src == "webcam" or src.lower().endswith((".mp4",".avi",".mov",".mkv")):
        process_video(src)
    elif os.path.isdir(src):
        process_folder(src)
    elif os.path.isfile(src):
        process_image(src)
    else:
        print(f"ERROR: fuente no encontrada: '{src}'")
