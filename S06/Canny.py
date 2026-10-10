# Importar librerías
import cv2
import matplotlib.pyplot as plt
import os

# Obtener la ruta de la imagen
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta_script, 'images.jpg')

print("Ruta imagen:", ruta_imagen)

# Leer la imagen
image = cv2.imread(ruta_imagen)

if image is None:
    raise FileNotFoundError(f"No se encontró {ruta_imagen}")

# Convertir de BGR a RGB para visualizar con Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar filtro Gaussiano para reducir ruido
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Detección de bordes con Canny
edges = cv2.Canny(
    blurred,
    threshold1=50,
    threshold2=150
)

# Mostrar resultados
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Bordes con Canny')
plt.axis('off')

plt.tight_layout()
plt.show()