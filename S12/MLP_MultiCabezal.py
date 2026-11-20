import torch
import torch.nn as nn

class DetectorMultiHeadMLP(nn.Module):
    def __init__(self, dim_caracteristicas=512, num_clases=5):
        super().__init__()
        # Tronco MLP compartido para refinar el vector de características de la región
        self.mlp_compartido = nn.Sequential(
            nn.Linear(dim_caracteristicas, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Cabezal 1: Clasificación (logits para las categorías de objetos)
        self.head_clasificacion = nn.Linear(128, num_clases)
        
        # Cabezal 2: Regresión de Bounding Box [x_min, y_min, x_max, y_max] normalizado en [0, 1]
        self.head_bbox = nn.Sequential(
            nn.Linear(128, 4),
            nn.Sigmoid()  # Restringe las coordenadas predichas al rango [0, 1] de la imagen
        )

    def forward(self, vector_caracteristicas):
        representacion = self.mlp_compartido(vector_caracteristicas)
        logits_clase = self.head_clasificacion(representacion)
        coordenadas_bbox = self.head_bbox(representacion)
        return logits_clase, coordenadas_bbox

# --- Simulación de entrenamiento Multi-Tarea ---
batch_size = 8
# Supongamos que un extractor visual entregó vectores de 512 dimensiones por imagen/región
features_visuales = torch.randn(batch_size, 512)

# Etiquetas reales (Ground Truth)
clases_gt = torch.randint(0, 5, (batch_size,))
cajas_gt = torch.tensor([
    [0.10, 0.15, 0.45, 0.60],
    [0.30, 0.20, 0.80, 0.90],
    [0.05, 0.05, 0.35, 0.40],
    [0.50, 0.50, 0.95, 0.95],
    [0.20, 0.30, 0.60, 0.70],
    [0.12, 0.18, 0.48, 0.55],
    [0.40, 0.10, 0.75, 0.45],
    [0.25, 0.25, 0.70, 0.80]
], dtype=torch.float32)

detector_mlp = DetectorMultiHeadMLP(dim_caracteristicas=512, num_clases=5)

# Funciones de pérdida separadas para cada cabezal
loss_fn_clase = nn.CrossEntropyLoss()
loss_fn_bbox = nn.SmoothL1Loss()  # Pérdida estándar en detección (ej. Faster R-CNN)

# Predicción y cálculo de pérdida combinada
pred_clases, pred_cajas = detector_mlp(features_visuales)
loss_c = loss_fn_clase(pred_clases, clases_gt)
loss_b = loss_fn_bbox(pred_cajas, cajas_gt)
loss_total = loss_c + (5.0 * loss_b)  # Lambda = 5.0 para equilibrar escalas

loss_total.backward()

print(f"Predicción clases (logits): {pred_clases.shape}")
print(f"Predicción cajas (bbox):    {pred_cajas.shape}")
print(f"Ejemplo de caja predicha (normalizada): {pred_cajas[0].detach().round(decimals=3).tolist()}")
print(f"Pérdida Clasificación: {loss_c.item():.4f} | Pérdida BBox: {loss_b.item():.4f}")