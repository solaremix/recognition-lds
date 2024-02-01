from PIL import Image
import pytesseract

# Setear el path hacia el ejecutable de tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar la imagen de medidor
image_path = r'img\2009542202402531653.jpg'
image = Image.open(image_path)

# Optical character recognition
text = pytesseract.image_to_string(image)

# Print the recognized text
print("Recognized text:")
print(text)
