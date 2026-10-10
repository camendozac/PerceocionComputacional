from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import corner_harris
import matplotlib.pyplot as plt
import numpy as np
import os

ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta_script, "balon.jpg")

image = imread(ruta_imagen)

image_gray = rgb2gray(image)

corners = corner_harris(image_gray, k=0.001)

image_copy = image.copy()

image_copy[corners > 0.01 * corners.max()] = [255, 0, 0]

plt.figure(figsize=(10, 6))
plt.imshow(image_copy)
plt.title("Detector de Esquinas de Harris")
plt.axis("off")
plt.show()
