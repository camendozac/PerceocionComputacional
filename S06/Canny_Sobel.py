# Importar librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta de la imagen
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta_script, 'images.jpg')

# Leer imagen
image = cv2.imread(ruta_imagen)

if image is None:
    raise FileNotFoundError(f"No se encontró {ruta_imagen}")

# Convertir a RGB para visualización
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Suavizado Gaussiano
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ----------------------------
# SOBEL
# ----------------------------
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# ----------------------------
# CANNY
# ----------------------------
canny = cv2.Canny(blurred, 100, 200)

# ----------------------------
# VISUALIZACIÓN
# ----------------------------
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Escala de Grises")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(sobel_magnitude, cmap="gray")
plt.title("Magnitud Sobel")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(canny, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()