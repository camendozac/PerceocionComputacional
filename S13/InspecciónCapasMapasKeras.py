import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from keras import layers, models

# 1. Definir arquitectura CNN con diferentes configuraciones de Conv y Pooling
inputs = layers.Input(shape=(32, 32, 3), name="Imagen_RGB")

# Bloque 1: Convolución 'same' (mantiene 32x32) + MaxPooling (reduce a 16x16)
x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name="Conv_1")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="MaxPool_1")(x)

# Bloque 2: Convolución 'valid' (reduce de 16x16 a 14x14) + AveragePooling (reduce a 7x7)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', name="Conv_2")(x)
x = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="AvgPool_2")(x)

# Bloque 3: Global Average Pooling (colapsa 7x7x32 -> vector de 32 elementos)
x = layers.GlobalAveragePooling2D(name="GlobalAvgPool")(x)
outputs = layers.Dense(units=10, activation='softmax', name="Clasificador")(x)

modelo = models.Model(inputs=inputs, outputs=outputs, name="CNN_Arquitectura_Informe")

# 2. Imprimir tabla técnica de dimensiones y parámetros entrenables
modelo.summary()

# 3. Simular el paso de una imagen aleatoria (batch=1) y verificar formas de salida
imagen_prueba = np.random.rand(1, 32, 32, 3).astype(np.float32)
prediccion = modelo(imagen_prueba)
print(f"\nForma del tensor de salida final: {prediccion.shape}")