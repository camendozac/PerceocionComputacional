import cv2
import numpy as np
import os

from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ==========================================
# PARÁMETROS DEL DESCRIPTOR LBP
# ==========================================

RADIO = 3
PUNTOS = 8 * RADIO

# ==========================================
# FUNCIÓN PARA EXTRAER CARACTERÍSTICAS LBP
# ==========================================

def obtener_histograma_lbp(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo leer: {ruta}"
        )

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    lbp = local_binary_pattern(
        gris,
        PUNTOS,
        RADIO,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, PUNTOS + 3),
        range=(0, PUNTOS + 2)
    )

    hist = hist.astype("float")

    # Normalización
    hist /= (hist.sum() + 1e-7)

    return hist


# ==========================================
# RUTAS
# ==========================================

ruta_base = os.path.dirname(
    os.path.abspath(__file__)
)

carpeta_objetos = os.path.join(
    ruta_base,
    "objetos"
)

# Imagen de prueba dentro de la carpeta objetos
ruta_prueba = os.path.join(
    carpeta_objetos,
    "prueba.jpg"
)

# ==========================================
# CARGAR IMÁGENES DE REFERENCIA
# ==========================================

referencias = []

print("\nContenido de la carpeta objetos:\n")

for archivo in os.listdir(carpeta_objetos):

    print(archivo)

    nombre = archivo.lower()

    # Excluir imagen de prueba
    if nombre == "prueba.jpg":
        continue

    if nombre.endswith(
        (".jpg", ".jpeg", ".png")
    ):

        referencias.append(
            os.path.join(
                carpeta_objetos,
                archivo
            )
        )

print("\nImágenes de referencia encontradas:\n")

for archivo in referencias:
    print(os.path.basename(archivo))

print(
    f"\nTotal referencias: {len(referencias)}"
)

# ==========================================
# CREAR MODELO DE TEXTURA PROMEDIO
# ==========================================

histogramas = []

for archivo in referencias:

    hist = obtener_histograma_lbp(
        archivo
    )

    histogramas.append(hist)

modelo_textura = np.mean(
    histogramas,
    axis=0
)

# ==========================================
# ANALIZAR IMAGEN DE PRUEBA
# ==========================================

hist_prueba = obtener_histograma_lbp(
    ruta_prueba
)

# ==========================================
# COMPARAR TEXTURAS
# ==========================================

similitud = cosine_similarity(
    [modelo_textura],
    [hist_prueba]
)[0][0]

porcentaje = similitud * 100

# ==========================================
# RESULTADOS
# ==========================================

print("\nRESULTADO")
print("--------------------------------")

print(
    f"Similitud encontrada: {porcentaje:.2f}%"
)

if porcentaje >= 90:

    clasificacion = "LADRILLO"
    mensaje = "Coincidencia ALTA"

elif porcentaje >= 75:

    clasificacion = "POSIBLE LADRILLO"
    mensaje = "Coincidencia MEDIA"