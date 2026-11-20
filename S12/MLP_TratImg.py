import torch
import torch.nn as nn
import torch.optim as optim
# 1. Definición de la arquitectura MLP para imágenes de 28x28 (1 canal)
class MLPImagenes(nn.Module):
    def __init__(self, alto=28, ancho=28, canales=1, num_clases=10):
        super().__init__()
        dim_entrada = alto * ancho * canales  # 784 características por imagen
        
        self.red = nn.Sequential(
            nn.Flatten(),                         # Transforma [B, 1, 28, 28] -> [B, 784]
            nn.Linear(dim_entrada, 256),          # Capa oculta 1
            nn.ReLU(),                            # Activación no lineal
            nn.Dropout(p=0.2),                    # Regularización para evitar sobreajuste
            nn.Linear(256, 128),                  # Capa oculta 2
            nn.ReLU(),
            nn.Linear(128, num_clases)            # Capa de salida (logits por clase)
        )

    def forward(self, x):
        return self.red(x)
# 2. Simulación de un lote de 16 imágenes normalizadas en [-1, 1]
lote_imagenes = torch.randn(16, 1, 28, 28)
etiquetas_reales = torch.randint(0, 10, (16,))
modelo_mlp = MLPImagenes()
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo_mlp.parameters(), lr=1e-3)
# 3. Paso de entrenamiento (Forward + Loss + Backpropagation)
optimizador.zero_grad()
logits = modelo_mlp(lote_imagenes)
perdida = criterio(logits, etiquetas_reales)
perdida.backward()
optimizador.step()
print(f"Forma de entrada: {lote_imagenes.shape}")
print(f"Forma de salida (logits): {logits.shape}")
print(f"Pérdida inicial calculada: {perdida.item():.4f}")