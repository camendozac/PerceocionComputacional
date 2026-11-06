import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
nombre_archivo = 'imagen2.jpg'  # <-- Tu archivo debe llamarse así en la carpeta
ruta_completa = os.path.join(carpeta_actual, nombre_archivo)

if not os.path.exists(ruta_completa):
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_completa}")

img_bgr = cv2.imread(ruta_completa)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_filtrada = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

# Cambio de dominio al espacio HSV
img_hsv = cv2.cvtColor(img_filtrada, cv2.COLOR_RGB2HSV)

colores_hsv = {
    # Rosas Rojas: El rojo es complejo porque se divide en dos extremos del canal Hue
    'Rojo_Bajo_1': (np.array([0, 70, 50]), np.array([10, 255, 255])),
    'Rojo_Alto_2': (np.array([165, 70, 50]), np.array([179, 255, 255])),
    
    # Rosas Rosadas y Fucsias
    'Rosa_Morado': (np.array([140, 50, 50]), np.array([165, 255, 255])),
    
    # Rosas Amarillas y Anaranjadas
    'Amarillo_Naranja': (np.array([11, 80, 50]), np.array([34, 255, 255]))
}

mascara_global = np.zeros(img_hsv.shape[:2], dtype=np.uint8)

# Evaluamos cada rango de color y los sumamos con una operación OR lógica
for nombre, (bajo, alto) in colores_hsv.items():
    mascara_color = cv2.inRange(img_hsv, bajo, alto)
    mascara_global = cv2.bitwise_or(mascara_global, mascara_color)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mascara_limpia = cv2.morphologyEx(mascara_global, cv2.MORPH_CLOSE, kernel)
mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_OPEN, kernel)

contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_resultado = img_rgb.copy()
contador_flores = 0

for c in contornos:
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    
    if area > 200:
        if perimetro == 0: continue
        circularidad = (4 * np.pi * area) / (perimetro ** 2)
        
        if circularidad > 0.20:
            contador_flores += 1
            x, y, w, h = cv2.boundingRect(c)
            
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_resultado, f"Flor #{contador_flores}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(img_rgb)
ax[0].set_title('1. Imagen Original Cargada')
ax[0].axis('off')

ax[1].imshow(mascara_limpia, cmap='gray')
ax[1].set_title('2. Máscara Multicolor Unificada')
ax[1].axis('off')

ax[2].imshow(img_resultado)
ax[2].set_title(f'3. Detección de Precisión\nTotal Flores: {contador_flores}')
ax[2].axis('off')

plt.tight_layout()
plt.show()
