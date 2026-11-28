import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import ndimage
import tensorflow as tf
from keras import datasets, layers, models

# 1. Función auxiliar para normalizar un recorte individual a formato MNIST (28x28)
def formatear_digito_mnist(recorte):
    alto, ancho = recorte.shape
    if alto > ancho:
        nuevo_alto = 20
        nuevo_ancho = max(1, int(round((ancho * 20.0) / alto)))
    else:
        nuevo_ancho = 20
        nuevo_alto = max(1, int(round((alto * 20.0) / ancho)))
        
    img_20 = Image.fromarray(recorte).resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    arr_20 = np.array(img_20, dtype=np.float32)
    
    lienzo = np.zeros((28, 28), dtype=np.float32)
    pad_y = (28 - nuevo_alto) // 2
    pad_x = (28 - nuevo_ancho) // 2
    lienzo[pad_y:pad_y+nuevo_alto, pad_x:pad_x+nuevo_ancho] = arr_20
    
    cy, cx = ndimage.center_of_mass(lienzo)
    desplazamiento_y = np.round(14.0 - cy).astype(int)
    desplazamiento_x = np.round(14.0 - cx).astype(int)
    lienzo = ndimage.shift(lienzo, shift=[desplazamiento_y, desplazamiento_x], mode='constant')
    
    lienzo = np.clip(lienzo / 255.0, 0.0, 1.0)
    return lienzo.reshape((1, 28, 28, 1)), lienzo

# 2. Segmentar múltiples dígitos (1 a 3 dígitos) de izquierda a derecha
def extraer_digitos_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).convert('L')
    if np.mean(img) > 127:
        img = ImageOps.invert(img)
    
    arr = np.array(img, dtype=np.float32)
    arr[arr < 35] = 0  # Eliminar ruido de fondo
    
    # Detectar cada trazo independiente (componentes conectados)
    mascara_binaria = arr > 0
    etiquetas, num_objetos = ndimage.label(mascara_binaria)
    objetos = ndimage.find_objects(etiquetas)
    
    candidatos = []
    for i, sl in enumerate(objetos):
        if sl is None:
            continue
        y_min, y_max = sl[0].start, sl[0].stop
        x_min, x_max = sl[1].start, sl[1].stop
        alto = y_max - y_min
        ancho = x_max - x_min
        
        # Filtrar puntos o manchas diminutas de ruido
        if alto < 8 or ancho < 3:
            continue
            
        # Aislar solo los píxeles pertenecientes a este componente
        recorte = np.where(etiquetas[sl] == (i + 1), arr[sl], 0)
        candidatos.append((x_min, recorte, (x_min, y_min, ancho, alto)))
    
    if not candidatos:
        raise ValueError("No se detectaron dígitos válidos en la imagen.")
    
    # Ordenar estrictamente de izquierda a derecha según x_min
    candidatos.sort(key=lambda item: item[0])
    
    digitos_procesados = []
    for _, recorte, bbox in candidatos:
        tensor, lienzo = formatear_digito_mnist(recorte)
        digitos_procesados.append((tensor, lienzo, bbox))
        
    return arr, digitos_procesados

# 3. Cargar MNIST y entrenar CNN (o cargar si ya fue guardada)
ruta_modelo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_mnist.keras")

if os.path.exists(ruta_modelo):
    print("Cargando modelo previamente entrenado...")
    cnn = models.load_model(ruta_modelo)
else:
    (x_train, y_train), _ = datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

    cnn = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Entrenando modelo CNN...")
    cnn.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
    cnn.save(ruta_modelo)

# 4. Cargar imagen, segmentar y predecir el número completo (0 - 100)
directorio_script = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(directorio_script, "mi_numero11.png")  # <-- Cambia al nombre de tu archivo

imagen_original, digitos = extraer_digitos_imagen(ruta)

predicciones = []
confianzas = []

for tensor, _, _ in digitos:
    probs = cnn(tensor, training=False).numpy()[0]
    predicciones.append(str(np.argmax(probs)))
    confianzas.append(np.max(probs) * 100)

numero_texto = "".join(predicciones)
valor_numerico = int(numero_texto)
confianza_promedio = np.mean(confianzas)

print(f"\n---> Número detectado: {valor_numerico} (Confianza promedio: {confianza_promedio:.2f}%)")
if not (0 <= valor_numerico <= 100):
    print(f"⚠️ Advertencia: El valor {valor_numerico} está fuera del rango esperado [0 - 100].")

# 5. Visualización de la detección y cada dígito segmentado
n = len(digitos)
fig, axes = plt.subplots(1, n + 1, figsize=(3.5 * (n + 1), 4))

# Imagen completa con recuadros de detección
axes[0].imshow(imagen_original, cmap='gray')
axes[0].set_title(f"Detectado: {valor_numerico}\n({confianza_promedio:.1f}% conf.)")
axes[0].axis('off')

for idx, (_, lienzo, (x, y, w, h)) in enumerate(digitos):
    rect = plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', linewidth=2)
    axes[0].add_patch(rect)
    
    axes[idx + 1].imshow(lienzo, cmap='gray')
    axes[idx + 1].set_title(f"Dígito {idx+1}: {predicciones[idx]}\n({confianzas[idx]:.1f}%)")
    axes[idx + 1].axis('off')

plt.tight_layout()
plt.show()