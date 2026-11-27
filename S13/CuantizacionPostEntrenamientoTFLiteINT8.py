import tensorflow as tf
import numpy as np

# 1. Crear un modelo de ejemplo entrenado
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 2. Configurar el convertidor con cuantización dinámica (FP32 -> INT8 en pesos)
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir y guardar el modelo consolidado ligero
modelo_tflite_cuantizado = converter.convert()
with open("modelo_consolidado.tflite", "wb") as f:
    f.write(modelo_tflite_cuantizado)

print(f"Tamaño del modelo consolidado: {len(modelo_tflite_cuantizado) / 1024:.2f} KB")

# 3. Inferencia usando el intérprete ligero de producción
interpreter = tf.lite.Interpreter(model_path="modelo_consolidado.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

imagen_entrada = np.random.rand(1, 28, 28, 1).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], imagen_entrada)
interpreter.invoke()

salida = interpreter.get_tensor(output_details[0]['index'])
print("Predicción en entorno Edge:", np.argmax(salida))