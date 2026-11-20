import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Cargar pesos preentrenados oficiales en el dataset COCO
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
modelo_deteccion = fasterrcnn_resnet50_fpn(weights=weights)
modelo_deteccion.eval()  # Modo evaluación (desactiva Dropout y congela BatchNorm)

# Simulación de una imagen tensor normalizada en rango [0, 1]
imagen_prueba = [torch.rand(3, 480, 640)]

with torch.no_grad():
    predicciones = modelo_deteccion(imagen_prueba)

# Inspección de la estructura de salida
print("Claves de salida del detector:", predicciones[0].keys())
# dict_keys(['boxes', 'labels', 'scores'])
print(f"Forma de cajas delimitadoras detectadas: {predicciones[0]['boxes'].shape}")