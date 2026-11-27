import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import ndimage
import tensorflow as tf
from keras import datasets, layers, models

# 1. Función de preprocesamiento estilo MNIST (Recorte + 20x20 + Centro de masa)
def preparar_imagen_estilo_mnist(ruta_imagen):
    img = Image.open(ruta_imagen).convert('L')
    if np.mean(img) > 127:
        img = ImageOps.invert(img)
    
    arr = np.array(img, dtype=np.float32)
    arr[arr < 30] = 0  # Limpiar ruido gris del fondo
    
    # Recortar el dígito (Bounding Box)
    filas = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not np.any(filas) or not np.any(cols):
        raise ValueError("La imagen está completamente vacía.")
    
    y_min, y_max = np.where(filas)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    recorte = arr[y_min:y_max+1, x_min:x_max+1]
    
    # Redimensionar manteniendo proporción a máximo 20x20 px
    alto, ancho = recorte.shape
    if alto > ancho:
        nuevo_alto = 20
        nuevo_ancho = max(1, int(round((ancho * 20.0) / alto)))
    else:
        nuevo_ancho = 20
        nuevo_alto = max(1, int(round((alto * 20.0) / ancho)))
        
    img_20 = Image.fromarray(recorte).resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    arr_20 = np.array(img_20, dtype=np.float32)
    
    # Pegar en lienzo negro 28x28 y centrar por Centro de Masa
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

# 2. Cargar MNIST y entrenar la CNN con Data Augmentation
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
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

# 3. Cargar tu imagen, preprocesar y predecir
ruta = "mi_numero.png"  # <-- Cambia esto si tu archivo tiene otro nombre
tensor_entrada, imagen_procesada = preparar_imagen_estilo_mnist(ruta)

probabilidades = cnn(tensor_entrada).numpy()[0]
numero_predicho = np.argmax(probabilidades)
confianza = np.max(probabilidades) * 100

print(f"\n---> Predicción final: {numero_predicho} (Confianza: {confianza:.2f}%)")

# 4. Mostrar resultado visual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(imagen_procesada, cmap='gray')
ax1.set_title(f"Imagen centrada (28x28)\nPredicción: {numero_predicho} ({confianza:.1f}%)")
ax1.axis('off')

ax2.bar(range(10), probabilidades * 100, color='steelblue')
ax2.set_xticks(range(10))
ax2.set_xlabel("Dígito (0-9)")
ax2.set_ylabel("Probabilidad (%)")
ax2.set_title("Salida Softmax")
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.show()