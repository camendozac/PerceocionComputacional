import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de la señal temporal
frecuencia_muestreo = 1000  # 1000 muestras por segundo (1 kHz)
tiempo = np.linspace(0, 1, frecuencia_muestreo, endpoint=False)  # 1 segundo de duración

# 2. Dominio Temporal: Creamos la señal original y el ruido
# Señal que nos interesa (ej. la voz de un comando o una vibración útil de 5 Hz)
senal_util = np.sin(2 * np.pi * 5 * tiempo)

# Ruido eléctrico constante (Interferencia de un motor o línea eléctrica a 50 Hz)
ruido_electrico = 1.5 * np.sin(2 * np.pi * 50 * tiempo)

# La señal real que recibe la computadora (Ambas mezcladas)
senal_contaminada = senal_util + ruido_electrico

# 3. Dominio Frecuencial: Aplicamos la FFT (Transformada Rápida de Fourier)
# Calculamos la transformada de la señal contaminada
fft_resultado = np.fft.fft(senal_contaminada)
# Obtenemos las frecuencias correspondientes a cada punto de la FFT
frecuencias = np.fft.fftfreq(frecuencia_muestreo, 1 / frecuencia_muestreo)

# Tomamos solo la mitad positiva del espectro (la FFT es simétrica)
mitad = frecuencia_muestreo // 2
frecuencias_positivas = frecuencias[:mitad]
# Calculamos la magnitud de la señal (valores absolutos)
magnitud = np.abs(fft_resultado[:mitad]) / mitad

# 4. Graficar los resultados para la clase
plt.figure(figsize=(12, 5))

# Gráfico 1: Dominio Temporal
plt.subplot(1, 2, 1)
plt.plot(tiempo, senal_contaminada, color='red', label='Señal Caótica (Mezclada)')
plt.plot(tiempo, senal_util, color='blue', alpha=0.7, linestyle='--', label='Señal Útil (Oculta)')
plt.title('Dominio Temporal: Amplitud vs Tiempo')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

# Gráfico 2: Dominio Frecuencial
plt.subplot(1, 2, 2)
plt.stem(frecuencias_positivas[:100], magnitud[:100], linefmt='black', markerfmt='D', basefmt=' ')
plt.title('Dominio Frecuencial: Espectro de Magnitud')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud / Fuerza')
plt.grid(True)

plt.tight_layout()
plt.show()
