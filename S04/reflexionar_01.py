import cv2
import numpy as np

def horizontal_reflection(imagen):
    altura, ancho, canales = imagen.shape
    imagen_reflejada = np.zeros((altura, ancho, canales), dtype=np.uint8)

    for i in range(altura):
        imagen_reflejada[i, :] = imagen[altura - i - 1, :]

    return imagen_reflejada

# 1. Cargar imagen
imagen = cv2.imread(
    r"D:\Programacion\Python\PercepcionComputacional\S04\imagen.jpg"
)

# 2. Ejecutar tu función
if imagen is None:
    print("ERROR: No se pudo cargar la imagen")
else:
    resultado = horizontal_reflection(imagen)

    cv2.imshow("Original", imagen)
    cv2.imshow("Reflejada", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()