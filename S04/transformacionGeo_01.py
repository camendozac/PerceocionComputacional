import cv2
import numpy as np

ruta = r"D:\Programacion\Python\PercepcionComputacional\S04\imagen.jpg"

imagen = cv2.imread(ruta)

# Verificar que se cargó correctamente
if imagen is None:
    raise FileNotFoundError(
        f"No se pudo cargar la imagen: {ruta}"
    )

alto, ancho = imagen.shape[:2]

# TRASLACIÓN
M = np.float32([
    [1, 0, 100],
    [0, 1, 50]
])

trasladada = cv2.warpAffine(
    imagen, M, (ancho, alto)
)

# ROTACIÓN
centro = (ancho // 2, alto // 2)

M_rotacion = cv2.getRotationMatrix2D(
    centro, 45, 1.0
)

rotada = cv2.warpAffine(
    imagen,
    M_rotacion,
    (ancho, alto)
)

# ESCALADO
escalada = cv2.resize(
    imagen,
    None,
    fx=0.5,
    fy=0.5
)

# VOLTEO HORIZONTAL
volteada = cv2.flip(imagen, 1)

# Mostrar resultados
cv2.imshow("Original", imagen)
cv2.imshow("Traslacion", trasladada)
cv2.imshow("Rotacion", rotada)
cv2.imshow("Escalado", escalada)
cv2.imshow("Volteo", volteada)

cv2.waitKey(0)
cv2.destroyAllWindows()