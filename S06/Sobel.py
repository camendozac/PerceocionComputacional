# Importar librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Obtener la ruta de la imagen
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta_script, 'images.jpg')

print("Ruta imagen:", ruta_imagen)

# Leer imagen
image = cv2.imread(ruta_imagen)

if image is None:
    raise FileNotFoundError(f"No se encontró {ruta_imagen}")

# Convertir BGR a RGB para visualización
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque Gaussiano
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Aplicar Sobel en X
sobel_x = cv2.Sobel(
    blurred,
    cv2.CV_64F,
    1, 0,
    ksize=3
)

# Aplicar Sobel en Y
sobel_y = cv2.Sobel(
    blurred,
    cv2.CV_64F,
    0, 1,
    ksize=3
)

# Magnitud del gradiente
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Convertir a uint8 para visualizar
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# Mostrar resultados
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(np.abs(sobel_x), cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Magnitud Sobel')
plt.axis('off')

plt.tight_layout()
plt.show()