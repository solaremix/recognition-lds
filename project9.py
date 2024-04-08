import cv2
import numpy as np
import pytesseract

# Cargar la imagen del medidor
file_path = 'images/1210816202402531954.jpg'
image = cv2.imread(file_path)

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mejorar el contraste utilizando ecualización de histograma
# Esto puede ayudar a mejorar la visibilidad de los dígitos
equ_image = cv2.equalizeHist(gray_image)

# Aplicar un desenfoque gaussiano para reducir el ruido
blurred_image = cv2.GaussianBlur(equ_image, (5, 5), 0)

# Binarización adaptativa para manejar diferentes condiciones de iluminación en la imagen
binary_image = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Encontrar contornos que podrían corresponder al área del medidor
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Suponemos que el área del medidor es el rectángulo más grande en la imagen
# Esto es solo una suposición y puede que no sea cierto en todos los casos
areas = [cv2.contourArea(c) for c in contours]
max_area_index = np.argmax(areas)
cnt = contours[max_area_index]

# Dibujar el contorno encontrado en la imagen original para visualización
contoured_image = image.copy()
cv2.drawContours(contoured_image, [cnt], -1, (0, 255, 0), 3)

# Guardar la imagen con contornos dibujados
output_path = 'segmentedImages/1210816202402531954.jpg'
cv2.imwrite(output_path, contoured_image)

output_path

# Configurar `pytesseract` para que se ejecute en el modo de solo dígitos
custom_config = r'--oem 3 --psm 6 outputbase digits'

# Encontrar el cuadro delimitador alrededor del contorno más grande
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.intp(box)

# Extraer la región delimitada por el cuadro delimitador
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")

# Coordenadas del cuadro delimitador después de que la imagen haya sido enderezada
dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")

# Matriz de transformación
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Aplicar transformación de perspectiva
warped = cv2.warpPerspective(gray_image, M, (width, height))

# Aplicar OCR en la imagen transformada
extracted_digits = pytesseract.image_to_string(warped, config=custom_config)

# Limpiar el texto extraído para eliminar cualquier caracter no deseado
extracted_digits = ''.join(filter(str.isdigit, extracted_digits))

print(extracted_digits)