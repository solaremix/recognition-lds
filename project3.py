import cv2
import numpy as np
import pytesseract

# Función para preprocesar la imagen
def preprocess_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Recortar la región de la imagen que contiene los dígitos
    height, width, _ = image.shape
    top_margin = 50  # Margen superior para evitar la fecha y la hora
    roi = image[top_margin:height, :]

    # Convertir la región recortada a escala de grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral adaptativo
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary

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
