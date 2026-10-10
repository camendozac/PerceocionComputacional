import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Carpeta donde se encuentra este archivo .py
carpeta = Path(__file__).parent

# Ruta completa de la imagen
ruta = carpeta / "coins.jpg"

print("Buscando imagen en:", ruta)

# Leer imagen
image = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)

# Comprobar que se cargó correctamente
if image is None:
    raise FileNotFoundError(
        f"No se encontró la imagen: {ruta}"
    )

# Imagen original
plt.figure(1)
plt.imshow(image, cmap="gray")
plt.title("Imagen original")
plt.axis("off")

# Umbralización de Otsu
umbral, otsu = cv2.threshold(
    image,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Resultado
plt.figure(2)
plt.imshow(otsu, cmap="gray")
plt.title("Segmentación mediante Otsu")
plt.axis("off")

print("Umbral Otsu:", umbral)

plt.show()