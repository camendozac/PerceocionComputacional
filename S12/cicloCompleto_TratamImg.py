import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.v2 as T

# 1. Transformaciones diferenciadas: Entrenamiento (con Augmentation) vs. Validación (determinista)
transform_train = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = T.Compose([
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Simulación de datasets de imágenes RGB (64x64) para 4 clases
X_train = torch.rand(80, 3, 64, 64)
y_train = torch.randint(0, 4, (80,))
X_val = torch.rand(20, 3, 64, 64)
y_val = torch.randint(0, 4, (20,))

loader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
loader_val = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

# 3. Modelo CNN compacto
modelo = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Dropout(0.3),
    nn.Linear(16 * 32 * 32, 4)
)

criterio = nn.CrossEntropyLoss()
optimizador = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="min", factor=0.5, patience=2)

# 4. Bucle de Entrenamiento y Validación con Checkpointing
mejor_val_loss = float("inf")

for epoca in range(1, 4):
    # --- FASE DE ENTRENAMIENTO ---
    modelo.train()
    loss_train_acum = 0.0
    for imagenes, etiquetas in loader_train:
        imagenes = transform_train(imagenes)
        optimizador.zero_grad()
        salidas = modelo(imagenes)
        loss = criterio(salidas, etiquetas)
        loss.backward()
        optimizador.step()
        loss_train_acum += loss.item() * imagenes.size(0)
    
    loss_train_media = loss_train_acum / len(loader_train.dataset)

    # --- FASE DE VALIDACIÓN ---
    modelo.eval()
    loss_val_acum = 0.0
    aciertos = 0
    with torch.no_grad():
        for imagenes, etiquetas in loader_val:
            imagenes = transform_val(imagenes)
            salidas = modelo(imagenes)
            loss = criterio(salidas, etiquetas)
            loss_val_acum += loss.item() * imagenes.size(0)
            predicciones = salidas.argmax(dim=1)
            aciertos += (predicciones == etiquetas).sum().item()

    loss_val_media = loss_val_acum / len(loader_val.dataset)
    exactitud_val = (aciertos / len(loader_val.dataset)) * 100
    scheduler.step(loss_val_media)

    print(f"Época {epoca:02d} | Train Loss: {loss_train_media:.4f} | "
          f"Val Loss: {loss_val_media:.4f} | Val Acc: {exactitud_val:.1f}%")

    if loss_val_media < mejor_val_loss:
        mejor_val_loss = loss_val_media
        # torch.save(modelo.state_dict(), "mejor_modelo_vision.pth")