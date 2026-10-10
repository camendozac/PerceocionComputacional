from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import (
    corner_harris,
    corner_peaks,
    corner_subpix
)
import matplotlib.pyplot as pylab
import os

ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(ruta_script, "balon.jpg")

image = imread(ruta_imagen)

image_gray = rgb2gray(image)

coordinates = corner_harris(image_gray, k=0.001)

corner_coordinates = corner_peaks(
    coordinates,
    min_distance=5
)

coordinates_subpix = corner_subpix(
    image_gray,
    corner_coordinates,
    window_size=11
)

pylab.figure(figsize=(12, 10))

pylab.subplot(211)
pylab.imshow(coordinates, cmap='inferno')

pylab.plot(
    coordinates_subpix[:,1],
    coordinates_subpix[:,0],
    'r.',
    markersize=5,
    label='Subpixel'
)

pylab.legend()
pylab.axis('off')

pylab.subplot(212)
pylab.imshow(image)

pylab.plot(
    corner_coordinates[:,1],
    corner_coordinates[:,0],
    'bo',
    markersize=4
)

pylab.plot(
    coordinates_subpix[:,1],
    coordinates_subpix[:,0],
    'r+',
    markersize=8
)

pylab.axis('off')
pylab.tight_layout()
pylab.show()