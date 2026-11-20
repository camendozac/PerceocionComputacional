import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

# 1. Definición de un pipeline de preprocesamiento y Data Augmentation
pipeline_tratamiento = T.Compose([
    T.ToImage(),                                      # Convierte entrada a tensor de imagen
    T.Resize((224, 224), antialias=True),             # Estandariza resolución espacial
    T.RandomHorizontalFlip(p=0.5),                    # Aumento de datos: espejo horizontal
    T.ColorJitter(brightness=0.2, contrast=0.2),      # Variación fotométrica
    T.ToDtype(torch.float32, scale=True),             # Escala píxeles de [0, 255] a [0.0, 1.0]
    T.Normalize(mean=[0.485, 0.456, 0.406],           # Normalización estadística por canal RGB
                std=[0.229, 0.224, 0.225])
])

# Simulación de una imagen sintética RGB de 300x400 en formato uint8 [0, 255]
imagen_cruda = torch.randint(0, 256, (3, 300, 400), dtype=torch.uint8)
imagen_procesada = pipeline_tratamiento(imagen_cruda)

print(f"Tensor original:  {imagen_cruda.shape}, dtype: {imagen_cruda.dtype}")
print(f"Tensor procesado: {imagen_procesada.shape}, dtype: {imagen_procesada.dtype}")

# 2. Aplicación de un filtro espacial (ej. detección de bordes verticales tipo Sobel)
filtro_vertical = torch.tensor([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
]).view(1, 1, 3, 3)  # [out_channels, in_channels, kH, kW]

conv_bordes = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
with torch.no_grad():
    conv_bordes.weight.copy_(filtro_vertical)

# Convertir imagen a 1 canal (escala de grises simulada) y extraer mapa de bordes
imagen_gris = imagen_procesada.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, 224, 224]
mapa_bordes = torch.relu(conv_bordes(imagen_gris))

print(f"Mapa de características resultante: {mapa_bordes.shape}")