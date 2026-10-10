import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image


def create_disk(radius):
    r = radius
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)

    y, x = np.meshgrid(y, x)

    return x*x + y*y <= r*r


# 1. Cargar coins.jpg en escala de grises
from pathlib import Path
from PIL import Image
import numpy as np

from pathlib import Path
from PIL import Image
import numpy as np

carpeta = Path(__file__).resolve().parent
ruta_imagen = carpeta / "coins.jpg"

print("Buscando imagen en:", ruta_imagen)

im_gray = np.array(
    Image.open(ruta_imagen).convert("L")
)

# 2. Convertir la imagen a binaria
# Puedes modificar 128 si fuera necesario
bw_coins = im_gray > 128

# 3. Crear elemento estructurante
radius = 2
strel = create_disk(radius)

# 4. Operaciones morfológicas
bw_eroded = ndimage.binary_erosion(
    bw_coins,
    structure=strel
)

bw_dilated = ndimage.binary_dilation(
    bw_coins,
    structure=strel
)

bw_opened = ndimage.binary_opening(
    bw_coins,
    structure=strel
)

bw_closed = ndimage.binary_closing(
    bw_coins,
    structure=strel
)

# 5. Mostrar resultados
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

imagenes = [
    (bw_coins, "Original binaria"),
    (bw_eroded, "Erosión"),
    (bw_dilated, "Dilatación"),
    (bw_opened, "Apertura"),
    (bw_closed, "Cierre")
]

for ax, (imagen, titulo) in zip(axes, imagenes):
    ax.imshow(imagen, cmap="gray")
    ax.set_title(titulo)
    ax.axis("off")

plt.tight_layout()
plt.show()