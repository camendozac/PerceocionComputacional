import torch
import torch.nn as nn

class BloqueConvolucional(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convolución 3x3 con padding=1 para mantener dimensiones espaciales
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # Normalización por lotes para estabilizar el entrenamiento
        self.bn = nn.BatchNorm2d(out_channels)
        # Activación no lineal
        self.relu = nn.ReLU(inplace=True)
        # Reducción espacial a la mitad
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Prueba del flujo de datos
entrada = torch.randn(1, 3, 64, 64)
bloque = BloqueConvolucional(in_channels=3, out_channels=16)
salida = bloque(entrada)

print(f"Dimensión de entrada: {entrada.shape}") # [1, 3, 64, 64]
print(f"Dimensión de salida:  {salida.shape}")  # [1, 16, 32, 32]