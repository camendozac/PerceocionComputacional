import numpy as np
import argparse
import cv2
import os

# Parseo de argumentos
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network (ONNX / OpenCV 5)')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--model", default="ssd_mobilenet_v1_12.onnx", help="path to ONNX model file")
parser.add_argument("--thr", default=0.4, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Ruta absoluta del modelo en la misma carpeta del script
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(carpeta_actual, args.model)

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"No se encontró el modelo ONNX en: {ruta_modelo}")

# Diccionario de clases COCO para ssd_mobilenet_v1_12
classNames = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

# Abrir archivo de video o cámara web (0)
if args.video:
    ruta_video = args.video if os.path.isabs(args.video) else os.path.join(carpeta_actual, args.video)
    cap = cv2.VideoCapture(ruta_video)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el archivo de video ni la cámara web.")

# Cargar modelo ONNX y obtener nombres de salida
net = cv2.dnn.readNetFromONNX(ruta_modelo)
nombres_salida = net.getUnconnectedOutLayersNames()

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:  # Termina el bucle si se acaba el video o falla la cámara
        break

    h, w = frame.shape[:2]

    # Preparar entrada para ssd_mobilenet_v1_12: RGB con forma (1, H, W, 3)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(rgb_frame, axis=0)

    net.setInput(input_tensor)
    salidas = net.forward(nombres_salida)

    # Identificar automáticamente cada tensor de salida
    boxes, classes, scores, count = None, None, None, 0
    for s in salidas:
        if s.ndim == 3 and s.shape[-1] == 4:
            boxes = s[0]
        elif s.ndim == 1 or s.size == 1:
            count = int(s.flatten()[0])
        elif s.ndim == 2:
            if np.all(s <= 1.0) and not np.all(np.equal(np.mod(s, 1), 0)):
                scores = s[0]
            else:
                classes = s[0].astype(int)

    # Dibujar detecciones
    for i in range(count):
        confidence = float(scores[i])
        if confidence > args.thr:
            class_id = classes[i]

            # Coordenadas vienen como [ymin, xmin, ymax, xmax] normalizadas
            ymin, xmin, ymax, xmax = boxes[i]
            xLeftBottom = int(xmin * w)
            yLeftBottom = int(ymin * h)
            xRightTop   = int(xmax * w)
            yRightTop   = int(ymax * h)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0), 2)

            nombre_clase = classNames.get(class_id, f"Clase {class_id}")
            label = f"{nombre_clase}: {confidence:.2f}"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                 (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                 (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            print(label)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Salir al presionar cualquier tecla (ej. ESC)
        break

cap.release()
cv2.destroyAllWindows()