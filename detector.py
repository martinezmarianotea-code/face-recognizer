"""
Módulo de detección de rostros.
Soporta YOLOv3 (OpenCV DNN) y YOLOv3u/YOLOv8 (Ultralytics).
Devuelve siempre una lista de bounding boxes: [(x, y, w, h), ...]
"""

import os
import threading
import cv2
import numpy as np
from config import (
    YOLO_VERSION,
    YOLO_V3_CFG, YOLO_V3_WEIGHTS,
    YOLO_VU_MODEL, DETECT_CONFIDENCE, NMS_THRESHOLD,
)


# ─── YOLOv3 clásico (OpenCV DNN) ──────────────────────────────────────────────

def _load_yolov3():
    if not os.path.exists(YOLO_V3_CFG) or not os.path.exists(YOLO_V3_WEIGHTS):
        raise FileNotFoundError(
            f"Modelos YOLOv3 no encontrados: '{YOLO_V3_CFG}' / '{YOLO_V3_WEIGHTS}'. "
            "Ejecutá 1_setup.py primero."
        )
    net = cv2.dnn.readNetFromDarknet(YOLO_V3_CFG, YOLO_V3_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


_yolov3_net = None
_yolov3_layers = None
_yolov3_lock = threading.Lock()


def detect_v3(frame):
    global _yolov3_net, _yolov3_layers
    if _yolov3_net is None:
        with _yolov3_lock:
            if _yolov3_net is None:
                _yolov3_net, _yolov3_layers = _load_yolov3()

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    _yolov3_net.setInput(blob)
    outputs = _yolov3_net.forward(_yolov3_layers)

    boxes, confidences = [], []
    for output in outputs:
        for det in output:
            scores = det[5:]
            conf = float(det[4])
            if conf < DETECT_CONFIDENCE:
                continue
            # YOLOv3-face tiene una sola clase (face)
            cx, cy, bw, bh = (det[0]*w, det[1]*h, det[2]*w, det[3]*h)
            x = int(cx - bw/2)
            y = int(cy - bh/2)
            boxes.append([x, y, int(bw), int(bh)])
            confidences.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, DETECT_CONFIDENCE, NMS_THRESHOLD)
    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append(tuple(boxes[i]))
    return result


# ─── YOLOv3u / YOLOv8 (Ultralytics) ──────────────────────────────────────────

_ultralytics_model = None
_ultralytics_lock = threading.Lock()


def detect_vu(frame):
    global _ultralytics_model
    if _ultralytics_model is None:
        with _ultralytics_lock:
            if _ultralytics_model is None:
                from ultralytics import YOLO as UltralyticsYOLO
                _ultralytics_model = UltralyticsYOLO(YOLO_VU_MODEL)

    results = _ultralytics_model(frame, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        # clase 0 = "person" en COCO; si usas un modelo face-only es clase 0 "face"
        if cls_id == 0 and conf >= DETECT_CONFIDENCE:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


# ─── API pública ──────────────────────────────────────────────────────────────

def detect_faces(frame):
    """
    Detecta rostros en `frame` (BGR numpy array).
    Retorna lista de (x, y, w, h).
    """
    if YOLO_VERSION == "v3":
        return detect_v3(frame)
    else:
        return detect_vu(frame)


def crop_face(frame, box, padding=10):
    """
    Recorta y devuelve la región del rostro con un poco de padding.
    """
    x, y, w, h = box
    H, W = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)
    return frame[y1:y2, x1:x2]
