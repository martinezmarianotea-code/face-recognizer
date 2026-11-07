"""
Configuración global del sistema de reconocimiento facial.
Cambia YOLO_VERSION a "v3" o "v3u" según lo que quieras usar.
"""

# ─── Detector ────────────────────────────────────────────────────────────────
YOLO_VERSION = "v3u"          # "v3" = YOLOv3 clásico (OpenCV DNN)
                               # "v3u" = YOLOv3u / YOLOv8  (Ultralytics)

# Rutas para YOLOv3 clásico (OpenCV DNN)
YOLO_V3_CFG     = "models/yolov3-face.cfg"
YOLO_V3_WEIGHTS = "models/yolov3-face.weights"

# Modelo para YOLOv3u / Ultralytics
YOLO_VU_MODEL   = "yolov8n.pt"   # se descarga automáticamente la primera vez

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_DIR     = "data/known_faces"   # data/known_faces/Persona/*.jpg
FACE_SIZE       = (100, 100)           # tamaño al que se normaliza cada rostro

# ─── Clasificador ─────────────────────────────────────────────────────────────
RECOGNIZER_MODEL = "models/face_recognizer.yml"
LABELS_FILE      = "models/labels.json"

# ─── Umbrales ─────────────────────────────────────────────────────────────────
DETECT_CONFIDENCE   = 0.45   # confianza mínima del detector YOLO
NMS_THRESHOLD       = 0.4    # supresión de no máximos (IoU): menor = menos cajas
RECOGNIZE_THRESHOLD = 110    # distancia LBPH: menor = más parecido
                             # sobre este valor se dice "Desconocido"
MIN_FACES_WARNING   = 5      # mínimo de imágenes por persona para entrenar
