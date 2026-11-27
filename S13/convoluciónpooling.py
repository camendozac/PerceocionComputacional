import numpy as np
import tensorflow as tf
import keras
from keras import datasets, layers, models

# 1. Imagen 5x5 de clase (mitad brillante = 10, mitad oscura = 0)
# Formato Keras: (batch_size, alto, ancho, canales)
imagen = np.array([
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0]
], dtype=np.float32).reshape((1, 5, 5, 1))

# 2. Definir modelo con 1 capa Convolucional + ReLU + MaxPooling
modelo_clase = models.Sequential([
    layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=1,
        padding='valid',  # Sin relleno (P = 0)
        activation='relu',
        use_bias=False,
        input_shape=(5, 5, 1)
    ),
    layers.MaxPooling2D(pool_size=(2, 2), strides=1)
])

# 3. Asignar manualmente el filtro detector de bordes verticales
# Formato filtro Keras: (alto, ancho, canales_entrada, num_filtros)
filtro_borde = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32).reshape((3, 3, 1, 1))

modelo_clase.layers[0].set_weights([filtro_borde])

# 4. Obtener salidas intermedias para proyectar en clase
salida_conv = modelo_clase.layers[0](imagen).numpy().squeeze()
salida_pool = modelo_clase.predict(imagen, verbose=0).squeeze()

print("Feature Map tras Conv2D + ReLU (3x3):\n", salida_conv)
print("\nSalida tras MaxPooling 2x2 (2x2):\n", salida_pool)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Evita bloqueos de oneDNN en TF 2.22-rc1

# print("1. Iniciando importaciones...", flush=True)
# import numpy as np
# import tensorflow as tf
# from keras import layers, models

# print("2. Creando imagen y modelo...", flush=True)
# imagen = np.array([
#     [10, 10, 10, 0, 0],
#     [10, 10, 10, 0, 0],
#     [10, 10, 10, 0, 0],
#     [10, 10, 10, 0, 0],
#     [10, 10, 10, 0, 0]
# ], dtype=np.float32).reshape((1, 5, 5, 1))

# modelo_clase = models.Sequential([
#     layers.Input(shape=(5, 5, 1)),
#     layers.Conv2D(
#         filters=1,
#         kernel_size=(3, 3),
#         strides=1,
#         padding='valid',
#         activation='relu',
#         use_bias=False
#     ),
#     layers.MaxPooling2D(pool_size=(2, 2), strides=1)
# ])

# filtro_borde = np.array([
#     [1, 0, -1],
#     [1, 0, -1],
#     [1, 0, -1]
# ], dtype=np.float32).reshape((3, 3, 1, 1))

# modelo_clase.layers[0].set_weights([filtro_borde])

# print("3. Calculando salidas...", flush=True)
# # Ejecución directa (Eager) en lugar de .predict() para evitar bloqueos
# salida_conv = modelo_clase.layers[0](imagen).numpy().squeeze()
# salida_pool = modelo_clase(imagen).numpy().squeeze()

# print("\nFeature Map tras Conv2D + ReLU (3x3):\n", salida_conv)
# print("\nSalida tras MaxPooling 2x2 (2x2):\n", salida_pool)