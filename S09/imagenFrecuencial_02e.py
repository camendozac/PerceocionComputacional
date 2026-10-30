import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('D:/Programacion/Python/PercepcionComputacional/S09/imgFrec_01.jpg', 
                    cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(imagen), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft) 

magnitud = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
espectro_magnitud = 20 * np.log(magnitud + 1)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(imagen, cmap='gray')
plt.title('Dominio Espacial (Original)'), plt.axis('off')
plt.subplot(122), plt.imshow(espectro_magnitud, cmap='gray')
plt.title('Dominio Frecuencial (Espectro)'), plt.axis('off')
plt.tight_layout()
plt.show()
