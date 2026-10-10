import cv2
import matplotlib.pyplot as plt
from pathlib import Path

ruta = Path(__file__).parent / "coins.jpg"

# Leer imagen original a color
image = cv2.imread(str(ruta))

if image is None:
    raise FileNotFoundError(f"No se encontró la imagen: {ruta}")

# OpenCV usa BGR, Matplotlib usa RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir a gris únicamente para Otsu
gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Umbralización de Otsu
umbral, otsu = cv2.threshold(
    gris,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Imagen original a color
plt.figure(1)
plt.imshow(image_rgb)
plt.title("Imagen original")
plt.axis("off")

# Resultado de Otsu
plt.figure(2)
plt.imshow(otsu, cmap="gray")
plt.title("Segmentación con Otsu")
plt.axis("off")

plt.show()

print("Umbral Otsu:", umbral)