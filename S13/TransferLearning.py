import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2

# 1. Cargar modelo base preentrenado sin la capa de clasificación superior
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Fase 1: Feature Extraction (congelar pesos base)
base_model.trainable = False

modelo = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # Ejemplo: clasificar 3 tipos de fallas industriales
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# (Aquí se entrenaría el clasificador durante unas épocas)

# 3. Fase 2: Fine-Tuning (descongelar las últimas 20 capas con tasa de aprendizaje baja)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
               loss='categorical_crossentropy',
               metrics=['accuracy'])