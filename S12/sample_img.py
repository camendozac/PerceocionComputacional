import numpy as np
import argparse
import cv2
import os

# Parseo de argumentos
parser = argparse.ArgumentParser(description='SSD MobileNet v1 (ONNX / OpenCV 5)')
parser.add_argument("--image", default="img.jpeg", help="path to image file")
parser.add_argument("--model", default="ssd_mobilenet_v1_12.onnx", help="path to ONNX model")
parser.add_argument("--thr", default=0.4, type=float, help="confidence threshold")
args = parser.parse_args()

# Rutas absolutas basadas en la ubicación del script
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(carpeta_actual, args.model)
ruta_imagen = os.path.join(carpeta_actual, args.image)

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")
if not os.path.exists(ruta_imagen):
    raise FileNotFoundError(f"No se encontró la imagen en: {ruta_imagen}")

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

# Cargar el modelo ONNX
net = cv2.dnn.readNetFromONNX(ruta_modelo)

# Cargar imagen
frame = cv2.imread(ruta_imagen)
if frame is None:
    raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")

h, w = frame.shape[:2]

# Preparar entrada: ssd_mobilenet_v1_12 espera RGB con forma (1, H, W, 3) en uint8
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
input_tensor = np.expand_dims(rgb_frame, axis=0)

net.setInput(input_tensor)

# Obtiene los nombres reales de salida que OpenCV 5 asignó al modelo
nombres_salida = net.getUnconnectedOutLayersNames()
salidas = net.forward(nombres_salida)

# Identifica automáticamente cada tensor por su forma (shape)
boxes, classes, scores, count = None, None, None, 0
for s in salidas:
    if s.ndim == 3 and s.shape[-1] == 4:          # (1, N, 4) -> cajas
        boxes = s[0]
    elif s.ndim == 1 or s.size == 1:              # (1,) -> número de detecciones
        count = int(s.flatten()[0])
    elif s.ndim == 2:                             # (1, N) -> clases (enteros) o puntajes (decimales <= 1.0)
        if np.all(s <= 1.0) and not np.all(np.equal(np.mod(s, 1), 0)):
            scores = s[0]
        else:
            classes = s[0].astype(int)

frame_copy = frame.copy()

# Primer bucle: dibujar rectángulos semitransparentes
for i in range(count):
    confidence = float(scores[i])
    if confidence > args.thr:
        # OJO: En este modelo las coordenadas vienen como [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = boxes[i]
        xLeftBottom_ = int(xmin * w)
        yLeftBottom_ = int(ymin * h)
        xRightTop_   = int(xmax * w)
        yRightTop_   = int(ymax * h)

        cv2.rectangle(frame_copy, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
                      (0, 255, 0), -1)

opacity = 0.3
cv2.addWeighted(frame_copy, opacity, frame, 1 - opacity, 0, frame)

# Segundo bucle: dibujar bordes negros y etiquetas
for i in range(count):
    confidence = float(scores[i])
    if confidence > args.thr:
        class_id = classes[i]
        ymin, xmin, ymax, xmax = boxes[i]
        xLeftBottom_ = int(xmin * w)
        yLeftBottom_ = int(ymin * h)
        xRightTop_   = int(xmax * w)
        yRightTop_   = int(ymax * h)

        cv2.rectangle(frame, (xLeftBottom_, yLeftBottom_), (xRightTop_, yRightTop_),
                      (0, 0, 0), 2)

        nombre_clase = classNames.get(class_id, f"Clase {class_id}")
        label = f"{nombre_clase}: {confidence:.2f}"
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 1)

        y_text = max(yLeftBottom_, labelSize[1])
        cv2.rectangle(frame, (xLeftBottom_, y_text - labelSize[1]),
                             (xLeftBottom_ + labelSize[0], y_text + baseLine),
                             (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (xLeftBottom_, y_text),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
        print(label)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()