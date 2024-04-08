from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo preentrenado para la clasificación de dígitos
model = load_model('cnn_model/digit_classifier.h5')

# Función para preprocesar la imagen y mejorar su calidad
def preprocess_image(image_path, contrast_factor=2.0):
    try:
        # Cargar la imagen
        image = Image.open(image_path)
        
        # Convertir la imagen a escala de grises
        image = image.convert('L')
        
        # Mejorar el contraste de la imagen
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # Aplicar un filtro para mejorar la nitidez de la imagen
        image = image.filter(ImageFilter.SHARPEN)
        
        return image
    except IOError:
        print("Error: No se pudo abrir la imagen")
        return None

# Función para reconocer dígitos en la imagen
def recognize_digits(image):
    try:
        # Utilizar Tesseract para realizar OCR en la imagen y obtener el texto
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        
        # Filtrar solo los dígitos numéricos usando expresiones regulares
        recognized_digits = ''.join(filter(str.isdigit, text))
        
        return recognized_digits
    except pytesseract.TesseractError:
        print("Error: No se pudo reconocer los dígitos")
        return None

# Función para predecir dígitos utilizando el modelo CNN
def predict_digit(img):
    test_image = img.reshape(-1,28,28,1)
    return np.argmax(model.predict(test_image))

# Función para refinar la imagen del medidor y reconocer dígitos
def recognize_meter_digits(image_path):
    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_path)
    
    if preprocessed_image:
        # Reconocer dígitos utilizando Tesseract
        recognized_tesseract_digits = recognize_digits(preprocessed_image)
        
        # Cargar la imagen con OpenCV para análisis adicional si es necesario
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Lógica adicional para segmentación, filtrado y procesamiento de imagen si es necesario
        
        # Procesamiento de la imagen para reconocer dígitos con el modelo CNN
        # Esto puede requerir la segmentación de los dígitos y la inferencia del modelo en cada uno
        
        # Por ahora, supongamos que recognized_tesseract_digits es suficiente para los dígitos reconocidos
        return recognized_tesseract_digits
    else:
        return None

if __name__ == "__main__":
    # Ruta de la imagen del medidor de luz
    image_path = 'images/1210816202402531954.jpg'
    
    # Reconocer los dígitos en la imagen del medidor
    recognized_digits = recognize_meter_digits(image_path)
    
    if recognized_digits:
        # Imprimir los dígitos reconocidos en la consola
        print(f"Dígitos reconocidos en la imagen {image_path}: {recognized_digits}")
