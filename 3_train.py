"""
Paso 3 – Entrenar el clasificador LBPH sobre las caras recolectadas.

LBPH (Local Binary Pattern Histograms):
  - Divide la cara en regiones pequeñas
  - En cada región calcula el patrón binario local (relación de cada píxel con sus vecinos)
  - Genera un histograma de patrones por región → vector de características
  - Robusto a variaciones de iluminación (por eso el usuario mencionó "patrón en su matriz, iluminación")

Uso:
  python 3_train.py
"""

import os
import json
import tempfile
import shutil
import cv2
import numpy as np
from config import DATASET_DIR, RECOGNIZER_MODEL, LABELS_FILE, FACE_SIZE, MIN_FACES_WARNING
from detector import detect_faces, crop_face

print("\n" + "═"*50)
print("  ENTRENAMIENTO DEL CLASIFICADOR LBPH")
print("═"*50)

# ─── 1. Cargar imágenes y etiquetas ───────────────────────────────────────────
faces  = []   # lista de imágenes en escala de grises
labels = []   # lista de enteros (id de persona)
label_map = {}       # {id: nombre}
label_map_inv = {}   # {nombre: id}

if not os.path.isdir(DATASET_DIR):
    print(f"ERROR: no existe '{DATASET_DIR}'. Ejecuta primero 2_collect_faces.py")
    exit(1)

persons = sorted([d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d))])

if not persons:
    print("No hay personas en el dataset. Agrega imágenes con 2_collect_faces.py")
    exit(1)

idx = 0
for person in persons:
    person_dir = os.path.join(DATASET_DIR, person)
    imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if not imgs:
        print(f"  ⚠  {person}: sin imágenes — omitido")
        continue
    if len(imgs) < MIN_FACES_WARNING:
        print(f"  ⚠  {person}: solo {len(imgs)} imagen(es) (mínimo recomendado: {MIN_FACES_WARNING})")

    label_map[idx] = person
    label_map_inv[person] = idx
    print(f"  [{idx}] {person:20s}  →  {len(imgs)} imágenes")
    idx += 1

    for fname in imgs:
        path = os.path.join(person_dir, fname)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        # Detectar cara con YOLO (igual que en reconocimiento)
        boxes = detect_faces(img_bgr)
        if boxes:
            # Usar la caja más grande (cara principal)
            box = max(boxes, key=lambda b: b[2] * b[3])
            face = crop_face(img_bgr, box)
        else:
            # Sin detección: usar imagen completa como fallback
            face = img_bgr

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, FACE_SIZE)
        gray = cv2.equalizeHist(gray)
        faces.append(gray)
        labels.append(idx)

if not faces:
    print("\nNo se encontraron imágenes válidas. Verifica la carpeta del dataset.")
    exit(1)

print(f"\nTotal imágenes cargadas: {len(faces)}")
print(f"Personas únicas: {len(label_map)}")

# ─── 2. Crear y entrenar el reconocedor LBPH ──────────────────────────────────
# Parámetros:
#   radius    = radio del LBP (cuántos píxeles alrededor se comparan)
#   neighbors = número de puntos del círculo LBP
#   grid_x/y  = cuántas celdas horizontales/verticales divide la imagen
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)

print("\nEntrenando...")
recognizer.train(faces, np.array(labels))

# ─── 3. Guardar modelo y etiquetas (escritura atómica) ────────────────────────
os.makedirs("models", exist_ok=True)
models_dir = os.path.dirname(os.path.abspath(RECOGNIZER_MODEL))

# Guardar modelo en archivo temporal y renombrar (evita corrupción si falla a mitad)
tmp_model_fd, tmp_model_path = tempfile.mkstemp(dir=models_dir, suffix=".yml")
os.close(tmp_model_fd)
try:
    recognizer.save(tmp_model_path)
    shutil.move(tmp_model_path, RECOGNIZER_MODEL)
except Exception as e:
    os.unlink(tmp_model_path)
    print(f"\nERROR al guardar modelo: {e}")
    exit(1)

tmp_labels_fd, tmp_labels_path = tempfile.mkstemp(dir=models_dir, suffix=".json")
try:
    with os.fdopen(tmp_labels_fd, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)
    shutil.move(tmp_labels_path, LABELS_FILE)
except Exception as e:
    os.unlink(tmp_labels_path)
    print(f"\nERROR al guardar etiquetas: {e}")
    exit(1)

print(f"\nModelo guardado en:   {RECOGNIZER_MODEL}")
print(f"Etiquetas guardadas:  {LABELS_FILE}")
print("\nEntrenamiento completado. Ejecuta ahora 4_recognize.py")
