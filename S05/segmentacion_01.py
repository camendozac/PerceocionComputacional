import cv2
import numpy as np

# Cámara de seguridad
camara = cv2.VideoCapture(0)

# Modelo simple del fondo
detector_fondo = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=40,
    detectShadows=True
)

# Elemento estructurante para operaciones morfológicas
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (5, 5)
)

while True:
    ret, frame = camara.read()

    if not ret:
        print("No se pudo obtener imagen de la cámara.")
        break

    # 1. Segmentación inicial
    mascara = detector_fondo.apply(frame)

    # 2. Umbralización
    _, mascara = cv2.threshold(
        mascara,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # 3. Operaciones morfológicas
    # Apertura: elimina pequeños puntos de ruido
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Cierre: rellena pequeños espacios
    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # 4. Detección de contornos
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objetos = 0

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        # Evita considerar ruido como un objeto
        if area > 1500:
            objetos += 1

            x, y, w, h = cv2.boundingRect(contorno)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Movimiento detectado",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.putText(
        frame,
        f"Objetos detectados: {objetos}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sistema de seguridad", frame)
    cv2.imshow("Mascara segmentada", mascara)

    # Presionar Q para terminar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()

