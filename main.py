import os.path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Función para preprocesar la imagen


def preprocess_image(image_path):

    # Cargar la imagen
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image = Image.open(os.path.join(
        script_dir, image_path))

    # Convertir la imagen a escala de grises
    image = image.convert('L')

    # Mejorar el contraste de la imagen
    enhancer = ImageEnhance.Contrast(image)
    # Ajustar el factor de mejora según sea necesario
    image = enhancer.enhance(2)

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
    image_path = "img\\186477020240252736 (7).jpg"

    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_path)

    # Reconocer dígitos en la imagen preprocesada
    recognized_digits = recognize_digits(preprocessed_image)

    # Imprimir los dígitos reconocidos en la consola
    print(f"Los dígitos de la imagen {image_path} son: {recognized_digits}")
