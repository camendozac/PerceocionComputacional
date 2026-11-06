import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
nombre_archivo = 'imagen2.jpg'
ruta_completa = os.path.join(carpeta_actual, nombre_archivo)

if not os.path.exists(ruta_completa):
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_completa}\n"
                            f"Verifica que el nombre tenga las mismas mayúsculas/minúsculas y la extensión correcta.")

img_bgr = cv2.imread(ruta_completa)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_suavizada = cv2.GaussianBlur(img_rgb, (7, 7), 0)

img_hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_RGB2HSV)

rango_inferior = np.array([15, 80, 80])
rango_superior = np.array([35, 255, 255])

mascara = cv2.inRange(img_hsv, rango_inferior, rango_superior)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mascara_limpia = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)

contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_resultado = img_rgb.copy()
contador_flores = 0

for c in contornos:
    # Filtro por área para descartar ruidos pequeños
    area = cv2.contourArea(c)
    if area > 150:  
        contador_flores += 1
        
        # Coordenadas del rectángulo delimitador
        x, y, w, h = cv2.boundingRect(c)
        
        # Dibujar la caja verde sobre la flor
        cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 3)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax = axes.ravel()

ax[0].imshow(img_rgb)
ax[0].set_title('1. Imagen Original')
ax[0].axis('off')

ax[1].imshow(mascara_limpia, cmap='gray')
ax[1].set_title('2. Máscara Binaria Filtrada')
ax[1].axis('off')

ax[2].imshow(img_resultado)
ax[2].set_title(f'3. Detección Final\nFlores encontradas: {contador_flores}')
ax[2].axis('off')

plt.tight_layout()
plt.show()
