import torch
import torchvision.ops as ops
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

# --- PARTE A: Fundamento matemático de IoU y Non-Maximum Suppression (NMS) ---
# Cajas simuladas en formato [x1, y1, x2, y2] para un mismo objeto
cajas_predichas = torch.tensor([
    [50.0,  50.0, 200.0, 200.0],  # Caja 0: alta confianza
    [55.0,  52.0, 198.0, 205.0],  # Caja 1: redundante (alto solapamiento con Caja 0)
    [300.0, 300.0, 420.0, 420.0]  # Caja 2: objeto distinto
])
confianzas = torch.tensor([0.92, 0.78, 0.88])

# Cálculo de matriz IoU entre todas las cajas
matriz_iou = ops.box_iou(cajas_predichas, cajas_predichas)
print("Matriz IoU entre cajas:\n", matriz_iou.round(decimals=2))

# Aplicar NMS con umbral IoU = 0.5 (elimina la Caja 1 por redundancia)
indices_conservados = ops.nms(cajas_predichas, confianzas, iou_threshold=0.5)
print("Índices conservados tras NMS:", indices_conservados.tolist())  # [0, 2]

# --- PARTE B: Detección de objetos con arquitectura One-Stage (FCOS) ---
pesos = FCOS_ResNet50_FPN_Weights.DEFAULT
detector = fcos_resnet50_fpn(weights=pesos, box_score_thresh=0.6)
detector.eval()

# Lote de prueba (1 imagen RGB de 480x640 normalizada en [0, 1])
imagen_lote = [torch.rand(3, 480, 640)]

with torch.no_grad():
    resultados = detector(imagen_lote)[0]

# Mapeo de índices de clase a etiquetas legibles de COCO
categorias_coco = pesos.meta["categories"]
etiquetas_detectadas = [categorias_coco[idx] for idx in resultados["labels"]]

print(f"\nObjetos detectados (confianza > 0.6): {len(resultados['boxes'])}")
for box, label, score in zip(resultados["boxes"], etiquetas_detectadas, resultados["scores"]):
    print(f" - {label}: {score:.2f} | Coordenadas: {box.round().tolist()}")