import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from keras import datasets, layers, models

# 1. Cargar y entrenar la CNN con MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

cnn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)

# 2. Función de preprocesamiento para imágenes de Paint
def preparar_imagen_paint(ruta_imagen):
    # Abrir imagen y convertir a escala de grises ('L')
    img = Image.open(ruta_imagen).convert('L')
    
    # Invertir colores si el fondo es blanco (MNIST usa fondo negro = 0, trazo blanco = 255)
    if np.mean(img) > 127:
        img = ImageOps.invert(img)
    
    # Redimensionar a 28x28 píxeles usando filtro suave (LANCZOS)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convertir a matriz NumPy, normalizar [0, 1] y dar forma (1, 28, 28, 1)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = img_array.reshape((1, 28, 28, 1))
    return img_tensor, img_array

# 3. Cargar tu dibujo y predecir
ruta = "mi_numero.png"
tensor_entrada, imagen_procesada = preparar_imagen_paint(ruta)

probabilidades = cnn(tensor_entrada).numpy()[0]
numero_predicho = np.argmax(probabilidades)
confianza = np.max(probabilidades) * 100

print(f"\n---> La CNN predice que dibujaste un: {numero_predicho} (Confianza: {confianza:.2f}%)")

# 4. Mostrar en pantalla cómo lo ve la red vs. las probabilidades de cada clase
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(imagen_procesada, cmap='gray')
ax1.set_title(f"Entrada 28x28 procesada\nPredicción: {numero_predicho} ({confianza:.1f}%)")
ax1.axis('off')

ax2.bar(range(10), probabilidades * 100, color='steelblue')
ax2.set_xticks(range(10))
ax2.set_xlabel("Dígito (Clases 0-9)")
ax2.set_ylabel("Probabilidad (%)")
ax2.set_title("Salida de la capa Softmax")
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.show()