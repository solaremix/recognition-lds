import cv2
import numpy as np
import pytesseract

def recognize_digits(image):
    recognized_digits = []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Filtrar contornos que se parecen a dígitos
        if 1 <= aspect_ratio <= 5 and cv2.contourArea(contour) > 100:
            digit_roi = image[y:y+h, x:x+w]
            digit_value = recognize_digit(digit_roi)
            recognized_digits.append(digit_value)
    
    return recognized_digits

def recognize_digit(digit_roi):
    # Aplicar umbral adaptativo en el dígito ROI
    _, thresholded_digit = cv2.threshold(digit_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Utilizar Tesseract para realizar OCR en el dígito ROI y obtener el texto
    digit_text = pytesseract.image_to_string(thresholded_digit, config='--psm 10 digits')
    
    # Filtrar los dígitos del texto reconocido
    recognized_digit = ''.join(c for c in digit_text if c.isdigit())
    
    # Retornar el dígito reconocido (o 'X' si no se reconoce)
    return recognized_digit if recognized_digit else "X"

if __name__ == "__main__":
    # Cargar la imagen
    image = cv2.imread("images/16981420240238232.jpg")
    
    # Convertir a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización
    _, image_threshold = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Aplicar desenfoque gaussiano para suavizar la imagen
    blur = cv2.GaussianBlur(image_threshold, (5, 5), 0)

    # Excluir la región superior (fecha y hora)
    top_region = 100  # Ajustar según la posición y tamaño de la región a excluir
    blur = blur[top_region:, :]

    # Realizar el reconocimiento de dígitos
    recognized_digits = recognize_digits(blur)

    print("Dígitos reconocidos:", recognized_digits)
