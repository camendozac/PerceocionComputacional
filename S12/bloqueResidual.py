import torch
import torch.nn as nn

class BloqueResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identidad = x  # Conexión de atajo (Skip Connection)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identidad  # Suma elemento a elemento antes de la activación final
        out = self.relu(out)
        return out

tensor_x = torch.randn(4, 64, 32, 32)
res_block = BloqueResidual(channels=64)
salida_res = res_block(tensor_x)
print(f"Salida del Bloque Residual: {salida_res.shape}") # Mantiene [4, 64, 32, 32]