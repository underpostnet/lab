import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread("tu_imagen.jpg")

# Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbralización
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

# Encontrar contornos
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Calcular y mostrar áreas
for cnt in contours:
    area = cv2.contourArea(cnt)
    print(f"Área del objeto: {area}")

# Visualizar (opcional)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow("Imagen con contornos", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
