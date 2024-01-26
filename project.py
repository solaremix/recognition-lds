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
