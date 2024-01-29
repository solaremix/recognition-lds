from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Función para preprocesar la imagen
def preprocess_image(image_path):
    # Cargar la imagen
    image = Image.open(image_path)
    
    # Convertir la imagen a escala de grises
    image = image.convert('L')
    
    # Mejorar el contraste de la imagen
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Ajustar el factor de mejora según sea necesario
    
    # Aplicar un filtro para mejorar la nitidez de la imagen
    image = image.filter(ImageFilter.SHARPEN)
    
    return image

# Función para reconocer dígitos en la imagen
def recognize_digits(image):
    # Utilizar Tesseract para realizar OCR en la imagen y obtener el texto
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    
    # Filtrar los dígitos del texto reconocido
    digits = ''.join(c for c in text if c.isdigit())
    
    return digits

if __name__ == "__main__":
    # Ruta de la imagen del medidor de luz
    image_path = '/Users/rodrigo/Desktop/LDS/recognition-lds/images/16981420240238232.jpg'
    
    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_path)
    
    # Reconocer dígitos en la imagen preprocesada
    recognized_digits = recognize_digits(preprocessed_image)
    
    # Imprimir los dígitos reconocidos en la consola
    print(f"Los dígitos de la imagen {image_path} son: {recognized_digits}")

# import cv2
# import pytesseract
# import numpy as np


# import cv2
# import pytesseract

# # Función para preprocesar la imagen
# def preprocess_image(image_path):
#     # Cargar la imagen
#     image = cv2.imread(image_path)

#     # Convertir la imagen a escala de grises
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Aplicar umbral adaptativo
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Aplicar un filtro para mejorar la nitidez de la imagen
#     sharpen_kernel = np.array([[-1, -1, -1],
#                                [-1, 9, -1],
#                                [-1, -1, -1]])
#     sharpened_image = cv2.filter2D(thresh, -1, sharpen_kernel)

#     return sharpened_image

# # Función para reconocer dígitos en la imagen
# def recognize_digits(image):
#     # Utilizar Tesseract para realizar OCR en la imagen y obtener el texto
#     text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

#     # Filtrar los dígitos del texto reconocido
#     digits = ''.join(c for c in text if c.isdigit())

#     return digits

# # Función para extraer la región de interés (ROI) de la imagen
# def extract_roi(image):
#     # Define las coordenadas de la región de interés (ROI) en la imagen
#     # Asegúrate de ajustar estas coordenadas según la posición de la fecha en tus imágenes
#     x, y, w, h = 0, 0, 100, 50

#     # Extrae la región de interés (ROI) de la imagen
#     roi = image[y:y+h, x:x+w]

#     return roi

# if __name__ == "__main__":
#     # Ruta de la imagen del medidor de luz
#     image_path = '/Users/rodrigo/Desktop/LDS/recognition-lds/images/16981420240238232.jpg'

#     # Preprocesar la imagen
#     preprocessed_image = preprocess_image(image_path)

#     # Extraer la región de interés (ROI) de la imagen preprocesada
#     roi = extract_roi(preprocessed_image)

#     # Reconocer dígitos en la región de interés (ROI)
#     recognized_digits = recognize_digits(roi)

#     # Imprimir los dígitos reconocidos en la consola
#     print(f"Los dígitos de la imagen {image_path} son: {recognized_digits}")
