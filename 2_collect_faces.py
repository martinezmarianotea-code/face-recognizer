"""
Paso 2 – Recolectar/registrar fotos de una persona.

Modos:
  a) Desde webcam  → python 2_collect_faces.py --name Persona --source webcam
  b) Desde carpeta → python 2_collect_faces.py --name Persona --source fotos/persona
  c) Desde video   → python 2_collect_faces.py --name Persona --source video.mp4

Las caras detectadas se guardan en:
  data/known_faces/<nombre>/face_001.jpg  ...
"""

import os
import cv2
import numpy as np
import argparse
from detector import detect_faces, crop_face
from config import DATASET_DIR, FACE_SIZE

# ─── Argumentos ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--name",   required=True,        help="Nombre de la persona")
parser.add_argument("--source", default="webcam",     help="'webcam', ruta de video o carpeta de imágenes")
parser.add_argument("--max",    type=int, default=50, help="Máximo de fotos a guardar (default 50)")
args = parser.parse_args()

# ─── Preparar directorio de salida ────────────────────────────────────────────
person_dir = os.path.join(DATASET_DIR, args.name)
os.makedirs(person_dir, exist_ok=True)

# Número de fotos ya existentes (para no sobreescribir)
existing = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
count = existing
print(f"\nRegistrando a: {args.name}")
print(f"Fotos previas: {existing}  |  Meta: {args.max} en total")
print("─" * 50)


def preprocess(face_bgr):
    """Convierte a escala de grises, redimensiona y ecualiza el histograma."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE)
    gray = cv2.equalizeHist(gray)   # normaliza iluminación
    return gray


def save_face(face_bgr):
    global count
    if count >= args.max:
        return False
    count += 1
    proc = preprocess(face_bgr)
    fname = os.path.join(person_dir, f"face_{count:03d}.jpg")
    cv2.imwrite(fname, proc)
    return True


def draw_info(frame, boxes, saved, total):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"{args.name}  {saved}/{total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


# ─── Modo WEBCAM / VIDEO ──────────────────────────────────────────────────────
def run_video(source):
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la fuente '{source}'")
        return

    frame_skip = 3   # procesar 1 de cada N frames (más rápido)
    frame_idx  = 0
    print("Espacio = captura manual | A = auto-captura | Q = salir")

    auto = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % frame_skip == 0:
                boxes = detect_faces(frame)
            else:
                boxes = []

            display = draw_info(frame.copy(), boxes, count, args.max)
            cv2.imshow("Recolectar caras", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                auto = not auto
                print("Auto-captura:", "ON" if auto else "OFF")
            elif key == ord(' '):
                for box in boxes:
                    face = crop_face(frame, box)
                    if face.size > 0:
                        saved = save_face(face)
                        if saved:
                            print(f"  Guardada cara {count}/{args.max}")

            if auto and boxes and frame_idx % (frame_skip * 5) == 0:
                for box in boxes:
                    face = crop_face(frame, box)
                    if face.size > 0:
                        save_face(face)

            if count >= args.max:
                print(f"\nMeta alcanzada: {args.max} fotos guardadas.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ─── Modo CARPETA DE IMÁGENES ─────────────────────────────────────────────────
def run_folder(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
              if f.lower().endswith(exts)]
    if not images:
        print(f"No se encontraron imágenes en '{folder}'")
        return

    for path in images:
        if count >= args.max:
            break
        frame = cv2.imread(path)
        if frame is None:
            continue
        boxes = detect_faces(frame)
        for box in boxes:
            face = crop_face(frame, box)
            if face.size > 0:
                save_face(face)
                print(f"  {os.path.basename(path)}  →  cara {count}")

    # Mostrar resumen visual de las últimas fotos guardadas
    samples = [os.path.join(person_dir, f)
               for f in sorted(os.listdir(person_dir))[-9:] if f.endswith(".jpg")]
    if samples:
        try:
            grid = _make_grid(samples, cols=3)
            cv2.imshow(f"Caras guardadas de {args.name}", grid)
            print("Presiona cualquier tecla para cerrar...")
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()


def _make_grid(paths, cols=3):
    imgs = [cv2.imread(p) for p in paths]
    imgs = [cv2.resize(i, FACE_SIZE) for i in imgs if i is not None]
    rows = [imgs[i:i+cols] for i in range(0, len(imgs), cols)]
    row_imgs = []
    for row in rows:
        while len(row) < cols:
            row.append(np.zeros((*FACE_SIZE, 3), np.uint8))
        row_imgs.append(np.hstack(row))
    return np.vstack(row_imgs)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    src = args.source
    if src == "webcam" or src.endswith((".mp4", ".avi", ".mov", ".mkv")):
        run_video(src)
    elif os.path.isdir(src):
        run_folder(src)
    else:
        print(f"ERROR: fuente no reconocida: '{src}'")

    saved_total = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
    print(f"\nTotal guardado en '{person_dir}': {saved_total} imágenes")
