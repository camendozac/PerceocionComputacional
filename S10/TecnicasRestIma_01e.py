#Instalar
#pip install numpy matplotlib scipy scikit-image
#Código
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import data, color, img_as_float
from skimage.restoration import wiener
imagen_original = img_as_float(color.rgb2gray(data.astronaut()))
tamanio_psf = 15
psf = np.zeros((tamanio_psf, tamanio_psf))
psf[tamanio_psf // 2, :] = 1.0
psf /= psf.sum()
imagen_borrosa = convolve2d(imagen_original, psf, mode='same', boundary='symm')
semilla_ruido = np.random.default_rng(42)
varianza_ruido = 0.002
ruido = semilla_ruido.normal(0, np.sqrt(varianza_ruido), imagen_borrosa.shape)
imagen_degradada = imagen_borrosa + ruido
G = np.fft.fft2(imagen_degradada)
H = np.fft.fft2(psf, s=imagen_degradada.shape)
H_estable = np.where(np.abs(H) < 1e-4, 1e-4, H)
F_inverso = G / H_estable
imagen_filtrado_inverso = np.real(np.fft.ifft2(F_inverso))
imagen_filtrado_inverso = np.clip(imagen_filtrado_inverso, 0, 1) 
restauracion_wiener = wiener(imagen_degradada, psf, balance=varianza_ruido)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes.ravel()
ax[0].imshow(imagen_original, cmap='gray')
ax[0].set_title('1. Imagen Original Limpia')
ax[0].axis('off')
ax[1].imshow(imagen_degradada, cmap='gray')
ax[1].set_title('2. Degradada (Desenfoque + Ruido)')
ax[1].axis('off')
ax[2].imshow(imagen_filtrado_inverso, cmap='gray')
ax[2].set_title('3. Intento de Filtrado Inverso\n(Falla por amplificación de ruido)')
ax[2].axis('off')
ax[3].imshow(restauracion_wiener, cmap='gray')
ax[3].set_title('4. Restauración con Filtro de Wiener\n(Resultado Óptimo)')
ax[3].axis('off')
plt.tight_layout()
plt.show()
