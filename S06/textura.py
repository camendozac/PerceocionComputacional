from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import os

ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta_script, "ladrillo.jpg")

imagen = imread(ruta_imagen)

gris = rgb2gray(imagen)
gris = img_as_ubyte(gris)

radio = 3
n_puntos = 8 * radio

lbp = local_binary_pattern(
    gris,
    n_puntos,
    radio,
    method="uniform"
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(imagen)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(lbp, cmap="gray")
plt.title("Textura LBP")
plt.axis("off")

plt.show()