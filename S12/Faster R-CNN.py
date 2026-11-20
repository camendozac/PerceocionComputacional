import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.ops as ops

# 1. Transformación sincronizada de imagen + Bounding Boxes
transform_deteccion = T.Compose([
    T.RandomHorizontalFlip(p=0.5),  # Invierte automáticamente tanto píxeles como coordenadas X
    T.ToDtype(torch.float32, scale=True)
])

# 2. Construcción de datos de ejemplo (2 imágenes con distinta cantidad de objetos)
img1 = torch.randint(0, 256, (3, 300, 300), dtype=torch.uint8)
boxes1 = tv_tensors.BoundingBoxes([[30, 40, 120, 150], [160, 170, 250, 260]],
                                  format="XYXY", canvas_size=(300, 300))
target1 = {"boxes": boxes1, "labels": torch.tensor([1, 2], dtype=torch.int64)}

img2 = torch.randint(0, 256, (3, 300, 300), dtype=torch.uint8)
boxes2 = tv_tensors.BoundingBoxes([[50, 50, 200, 220]],
                                  format="XYXY", canvas_size=(300, 300))
target2 = {"boxes": boxes2, "labels": torch.tensor([1], dtype=torch.int64)}

# Aplicar transformación sincronizada
img1_t, target1_t = transform_deteccion(img1, target1)
img2_t, target2_t = transform_deteccion(img2, target2)

imagenes_lote = [img1_t, img2_t]
targets_lote = [target1_t, target2_t]

# 3. Instanciar Faster R-CNN adaptado a 3 clases (0: fondo, 1: clase A, 2: clase B)
modelo_det = fasterrcnn_resnet50_fpn(weights=None, num_classes=3)
optimizador = torch.optim.SGD(modelo_det.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

# --- FASE DE ENTRENAMIENTO ---
modelo_det.train()
optimizador.zero_grad()

# En modo train(), devuelve un diccionario con las 4 pérdidas internas del detector
diccionario_perdidas = modelo_det(imagenes_lote, targets_lote)
perdida_total = sum(loss for loss in diccionario_perdidas.values())

perdida_total.backward()
optimizador.step()

print("Desglose de pérdidas de entrenamiento:")
for nombre, valor in diccionario_perdidas.items():
    print(f" - {nombre}: {valor.item():.4f}")
print(f"Pérdida total combinada: {perdida_total.item():.4f}\n")

# --- FASE DE VALIDACIÓN (Evaluación espacial mediante IoU) ---
modelo_det.eval()
with torch.no_grad():
    predicciones_val = modelo_det([img1_t])

cajas_pred = predicciones_val[0]["boxes"]
cajas_reales = target1_t["boxes"]

if len(cajas_pred) > 0:
    iou_validacion = ops.box_iou(cajas_pred, cajas_reales)
    print(f"Cajas predichas en validación: {len(cajas_pred)}")
    print(f"IoU máximo alcanzado respecto al Ground Truth: {iou_validacion.max().item():.4f}")
else:
    print("Validación ejecutada: 0 detecciones sobre el umbral en esta iteración inicial.")