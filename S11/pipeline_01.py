import time
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# ==============================================================================
# 1. ADQUISICIÓN Y PREPROCESAMIENTO DE SEÑALES VISUALES (OpenCV)
# ==============================================================================
print("[1/4] Cargando y preprocesando señales sensoriales...")

lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
X_raw = lfw_dataset.images
y = lfw_dataset.target
target_names = lfw_dataset.target_names

def preprocess_signal(images: np.ndarray, target_size=(64, 64)) -> list:
    """Normaliza resolución e iluminación mediante OpenCV."""
    processed_imgs = []
    for img in images:
        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        resized = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)
        processed_imgs.append(equalized)
    return processed_imgs

X_preprocessed = preprocess_signal(X_raw, target_size=(64, 64))

# ==============================================================================
# 2. EXTRACCIÓN DE CARACTERÍSTICAS (HOG) Y DIVISIÓN TRAIN / TEST
# ==============================================================================
print("[2/4] Extrayendo descriptores HOG...")

def extract_hog_features(images: list) -> np.ndarray:
    """Extrae el vector de características HOG usando scikit-image."""
    features = []
    for img in images:
        hog_vector = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )
        features.append(hog_vector)
    return np.array(features)

X_hog = extract_hog_features(X_preprocessed)
print(f"      Dimensionalidad original: {64 * 64} -> Vector HOG: {X_hog.shape[1]} características")

# ESTA ES LA LÍNEA QUE FALTABA EN TU ARCHIVO:
X_train, X_test, y_train, y_test = train_test_split(
    X_hog, y, test_size=0.25, random_state=42, stratify=y
)

# ==============================================================================
# 3. CONSTRUCCIÓN DEL PIPELINE: ESTANDARIZACIÓN + PCA + SVM
# ==============================================================================
print("[3/4] Entrenando Pipeline (StandardScaler -> PCA -> SVM)...")

perception_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, whiten=True, random_state=42)),
    ("svm", SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

t_start_train = time.perf_counter()
perception_pipeline.fit(X_train, y_train)
t_train = time.perf_counter() - t_start_train

n_components_kept = perception_pipeline.named_steps["pca"].n_components_
print(f"      PCA redujo el espacio de {X_hog.shape[1]} a {n_components_kept} componentes principales.")
print(f"      Tiempo de entrenamiento: {t_train:.3f} segundos")

# ==============================================================================
# 4. INFERENCIA Y REPORTE DE MÉTRICAS DE EVALUACIÓN
# ==============================================================================
print("\n[4/4] Evaluando rendimiento perceptual y latencia...")

t_start_inf = time.perf_counter()
y_pred = perception_pipeline.predict(X_test)
t_inf_total = time.perf_counter() - t_start_inf
fps = len(X_test) / t_inf_total

print(f"\n--- RENDIMIENTO COMPUTACIONAL ---")
print(f"Muestras evaluadas : {len(X_test)}")
print(f"Latencia por imagen: {(t_inf_total / len(X_test)) * 1000:.2f} ms ({fps:.1f} FPS)")

print(f"\n--- MÉTRICAS GLOBALES ---")
print(f"Accuracy Global    : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (Macro)   : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-Score (Weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\n--- REPORTE DETALLADO POR CLASE ---")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

print("--- MATRIZ DE CONFUSIÓN ---")
print(confusion_matrix(y_test, y_pred))