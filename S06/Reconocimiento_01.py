# Import librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
import os
import cv2

ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta_script, 'images.jpg')

print("Ruta imagen:", ruta_imagen)

image = cv2.imread(ruta_imagen)

if image is None:
    raise FileNotFoundError(f"No se encontró {ruta_imagen}")

# Convertir BGR a RGB (OpenCV carga en formato BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mostrar la imagen original
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Aplicar detección de bordes de Canny
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# Mostrar los resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()