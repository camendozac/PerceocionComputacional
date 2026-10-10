import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from pathlib import Path


def create_disk(radius):
    r = radius
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)

    y, x = np.meshgrid(y, x)

    return x*x + y*y <= r*r


# Ruta de la imagen
carpeta = Path(__file__).resolve().parent
ruta_imagen = carpeta / "coins.jpg"

print("Buscando imagen en:", ruta_imagen)

# 1. Cargar imagen ORIGINAL A COLOR
im_color = np.array(
    Image.open(ruta_imagen).convert("RGB")
)

# 2. Cargar imagen en escala de grises para procesarla
im_gray = np.array(
    Image.open(ruta_imagen).convert("L")
)

# 3. Convertir a imagen binaria
bw_coins = im_gray > 128

# 4. Crear elemento estructurante
radius = 2
strel = create_disk(radius)

# 5. Operaciones morfológicas
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

# 6. Mostrar resultados
fig, axes = plt.subplots(1, 5, figsize=(16, 5))

# Imagen original A COLOR
axes[0].imshow(im_color)
axes[0].set_title("Original")
axes[0].axis("off")

# Resultados binarios
resultados = [
    (bw_eroded, "Erosión"),
    (bw_dilated, "Dilatación"),
    (bw_opened, "Apertura"),
    (bw_closed, "Cierre")
]

for ax, (imagen, titulo) in zip(axes[1:], resultados):
    ax.imshow(imagen, cmap="gray")
    ax.set_title(titulo)
    ax.axis("off")

plt.tight_layout()
plt.show()