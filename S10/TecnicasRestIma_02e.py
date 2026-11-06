import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io, color, img_as_float

carpeta_actual = os.path.dirname(os.path.abspath(__file__))

nombre_archivo = "imagen.jpg"
ruta_completa = os.path.join(carpeta_actual, nombre_archivo)

if not os.path.exists(ruta_completa):
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_completa}\n"
                            f"Verifica que el nombre tenga las mismas mayúsculas/minúsculas y la extensión (.jpg o .png) correcta.")

imagen_original = img_as_float(color.rgb2gray(io.imread(ruta_completa)))

tamanio_kernel = 21
sigma = 5.0
x, y = np.mgrid[-tamanio_kernel//2 + 1:tamanio_kernel//2 + 1, -tamanio_kernel//2 + 1:tamanio_kernel//2 + 1]
psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
psf /= psf.sum()

imagen_borrosa = convolve2d(imagen_original, psf, mode='same', boundary='symm')

semilla_ruido = np.random.default_rng(42)
ruido = semilla_ruido.normal(0, np.sqrt(1e-5), imagen_borrosa.shape)
imagen_borrosa_con_ruido = imagen_borrosa + ruido

H = np.fft.fft2(psf, s=imagen_original.shape)
G_sin_ruido = np.fft.fft2(imagen_borrosa)
G_con_ruido = np.fft.fft2(imagen_borrosa_con_ruido)

F_inverso_puro = G_sin_ruido / H
imagen_restaurada_pura = np.real(np.fft.ifft2(F_inverso_puro))
imagen_restaurada_pura = np.clip(imagen_restaurada_pura, 0, 1)

alpha = 0.05  # Controla la tolerancia al ruido
H_truncada = np.where(np.abs(H) < alpha, H + alpha * np.sign(H), H)

F_inverso_truncado = G_con_ruido / H_truncada
imagen_restaurada_truncada = np.real(np.fft.ifft2(F_inverso_truncado))
imagen_restaurada_truncada = np.clip(imagen_restaurada_truncada, 0, 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes.ravel()

ax[0].imshow(imagen_original, cmap='gray')
ax[0].set_title('1. Imagen Original Cargada')
ax[0].axis('off')

ax[1].imshow(imagen_borrosa, cmap='gray')
ax[1].set_title('2. Imagen con Desenfoque Óptico')
ax[1].axis('off')

ax[2].imshow(imagen_restaurada_pura, cmap='gray')
ax[2].set_title('3. Filtrado Inverso Perfecto\n(Desenfoque puro sin ruido)')
ax[2].axis('off')

ax[3].imshow(imagen_restaurada_truncada, cmap='gray')
ax[3].set_title('4. Filtrado Inverso Truncado\n(Atenúa la catástrofe del ruido)')
ax[3].axis('off')

plt.tight_layout()
plt.show()
