import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Parámetros de simulación
fs = 1000  # Frecuencia de muestreo
t = np.arange(0, 4.0, 1/fs)  # Señal de 4 segundos

# Dos frecuencias muy próximas para forzar el límite de resolución
f1, f2 = 50.0, 55.0
senal_pura = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
ruido = np.random.normal(0, 1.5, len(t))
senal_ruidosa = senal_pura + ruido

# 2. Configuración de Welch con ventana de segmento corta (nperseg = 128)
# Produce más segmentos independientes para promediar -> Menos ruido, pero peor resolución.
f_corto, psd_corto = signal.welch(senal_ruidosa, fs, window='hann', nperseg=128, noverlap=64)

# 3. Configuración de Welch con ventana de segmento larga (nperseg = 1024)
# Produce menos segmentos -> Más picos ruidosos, pero con alta definición frecuencial.
f_largo, psd_largo = signal.welch(senal_ruidosa, fs, window='hann', nperseg=1024, noverlap=512)

# 4. Visualización comparativa
plt.figure(figsize=(12, 5))

plt.plot(f_corto, psd_corto, label='Segmento Corto (nperseg=128) - Suave pero sin resolución', color='orange', linewidth=2)
plt.plot(f_largo, psd_largo, label='Segmento Largo (nperseg=1024) - Ruidoso pero separa picos', color='blue', alpha=0.7)

plt.axvline(f1, color='red', linestyle='--', alpha=0.5, label='Frecuencias Reales (50 y 55 Hz)')
plt.axvline(f2, color='red', linestyle='--', alpha=0.5)

plt.title('Compromiso en el Método de Welch: Resolución vs Varianza')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad Espectral de Potencia')
plt.xlim(30, 80)  # Zoom en el rango de interés
plt.grid(True)
plt.legend()
plt.show()
