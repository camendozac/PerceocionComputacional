import cv2

imagen = cv2.imread("imagen.jpg", cv2.IMREAD_GRAYSCALE)
resultado = cv2.equalizeHist(imagen)

cv2.imshow("Original", imagen)
cv2.imshow("Ecualizada", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

imagen = cv2.imread("imagen.jpg", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

resultado = clahe.apply(imagen)

cv2.imshow("Original", imagen)
cv2.imshow("Correccion CLAHE", resultado)

cv2.waitKey(0)
cv2.destroyAllWindows()