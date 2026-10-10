import cv2
import numpy as np
from tkinter import Tk, filedialog

def horizontal_reflection(imagen):
    altura, ancho, canales = imagen.shape

    imagen_reflejada = np.zeros(
        (altura, ancho, canales),
        dtype=np.uint8
    )

    for i in range(altura):
        imagen_reflejada[i, :] = imagen[altura - i - 1, :]

    return imagen_reflejada


# Ocultar ventana principal de Tkinter
Tk().withdraw()

# Abrir ventana para seleccionar imagen
ruta = filedialog.askopenfilename(
    title="Seleccionar imagen",
    filetypes=[
        ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
        ("Todos los archivos", "*.*")
    ]
)

if ruta:
    imagen = cv2.imread(ruta)

    if imagen is not None:
        resultado = horizontal_reflection(imagen)

        cv2.imshow("Imagen original", imagen)
        cv2.imshow("Imagen reflejada", resultado)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se pudo leer la imagen.")
else:
    print("No seleccionaste ninguna imagen.")