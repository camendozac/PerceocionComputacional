# import tensorflow as tf
# import keras
# from keras import datasets, layers, models

# # 1. Cargar y normalizar el dataset MNIST (imágenes 28x28 en escala de grises)
# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
# x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

# # 2. Definir la arquitectura CNN
# cnn = models.Sequential([
#     # Bloque 1: Extracción de características simples (bordes/trazos)
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
#     layers.MaxPooling2D((2, 2)),  # Reduce de 28x28 a 14x14

#     # Bloque 2: Extracción de patrones complejos (curvas/bucles)
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2)),  # Reduce de 14x14 a 7x7

#     # Bloque 3: Clasificación
#     layers.Flatten(),                        # Vectoriza 7x7x64 = 3,136 valores
#     layers.Dense(64, activation='relu'),     # Capa densa intermedia
#     layers.Dense(10, activation='softmax')   # 10 probabilidades (dígitos 0 al 9)
# ])

# # 3. Mostrar resumen de capas y parámetros (ideal para analizar dimensiones en clase)
# cnn.summary()

# # 4. Compilar y entrenar
# cnn.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# cnn.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

# # 5. Evaluar en datos no vistos
# loss, acc = cnn.evaluate(x_test, y_test, verbose=0)
# print(f"\nPrecisión en test: {acc * 100:.2f}%")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models

# 1. Cargar y normalizar el dataset MNIST (dígitos escritos a mano 0-9)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

# 2. Definir la arquitectura CNN
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

# 3. Compilar y entrenar
cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
cnn.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

# 4. Evaluar precisión global
loss, acc = cnn.evaluate(x_test, y_test, verbose=0)
print(f"\nPrecisión en test (10,000 dígitos): {acc * 100:.2f}%")

# 5. Predecir las primeras 10 imágenes de prueba y mostrarlas visualmente
predicciones = cnn(x_test[:10]).numpy()

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    
    clase_predicha = np.argmax(predicciones[i])
    confianza = np.max(predicciones[i]) * 100
    clase_real = y_test[i]
    
    color = 'green' if clase_predicha == clase_real else 'red'
    plt.title(f"Real: {clase_real} | Pred: {clase_predicha}\n({confianza:.1f}%)", color=color)
    plt.axis('off')

plt.suptitle("Clasificación de Dígitos Escritos a Mano (Dataset MNIST)", fontsize=14)
plt.tight_layout()
plt.show()