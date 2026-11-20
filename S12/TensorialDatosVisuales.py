import torch
import torch.nn as nn

# Simulación de un lote de 8 imágenes RGB de 224x224 píxeles (Formato NCHW en PyTorch)
batch_size, channels, height, width = 8, 3, 224, 224
imagen_tensor = torch.randn(batch_size, channels, height, width)

print(f"Forma del tensor visual: {imagen_tensor.shape}")

# Comparación de parámetros: Capa Densa (MLP) vs. Capa Convolucional
entrada_aplanada = channels * height * width  # 150,528 características
capa_densa = nn.Linear(in_features=entrada_aplanada, out_features=64)
capa_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

params_densa = sum(p.numel() for p in capa_densa.parameters())
params_conv = sum(p.numel() for p in capa_conv.parameters())

print(f"Parámetros en Capa Densa: {params_densa:,}")         # 9,633,856 parámetros
print(f"Parámetros en Capa Convolucional: {params_conv:,}") # 1,792 parámetros