from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Función para preprocesar la imagen
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
        
        return text
    except pytesseract.TesseractError:
        print("Error: No se pudo reconocer los dígitos")
        return None

if __name__ == "__main__":
    # Ruta de la imagen del medidor de luz
    image_path = 'images/1974545202402531533.jpg'
    
    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_path)
    
    if preprocessed_image:
        # Reconocer dígitos en la imagen preprocesada
        recognized_text = recognize_digits(preprocessed_image)
        
        if recognized_text:
            # Imprimir el texto reconocido en la consola
            print(f"Texto reconocido en la imagen {image_path}: {recognized_text}")
