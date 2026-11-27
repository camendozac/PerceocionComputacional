import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Simulación de etiquetas reales y predicciones para 3 clases: 0=Sano, 1=Leve, 2=Crítico
y_real = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1, 2])
y_pred = np.array([0, 0, 1, 0, 2, 1, 2, 0, 1, 2])

print("Matriz de Confusión:")
print(confusion_matrix(y_real, y_pred))

print("\nReporte Detallado por Clase:")
print(classification_report(y_real, y_pred, target_names=['Sano', 'Leve', 'Crítico']))