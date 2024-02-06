import cv2
import pytesseract

# Función para extraer dígitos de una imagen de la región de interés (ROI)
def extract_digits_from_roi(roi_image_path):
    # Cargar la imagen de la región de interés (ROI)
    roi_image = cv2.imread(roi_image_path)

    # Convertir la imagen de la región de interés a escala de grises
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Aplicar cualquier preprocesamiento necesario, como umbralización u otras técnicas
    # Convertir la imagen a escala de grises

    # Aplicar suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Aplicar umbralización para resaltar la capa de vidrio/plástico
    _, threshold = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Utilizar Tesseract para realizar OCR en la imagen y obtener el texto
    text = pytesseract.image_to_string(gray_roi, lang='eng', config='--psm 6')

    # Filtrar los dígitos del texto reconocido
    digits = ''.join(c for c in text if c.isdigit())

    return digits

if __name__ == "__main__":
    # Ruta de la imagen de la región de interés (ROI)
    roi_image_path = "images/211955020240253205.jpg"

    # Extraer dígitos de la imagen de la región de interés
    extracted_digits = extract_digits_from_roi(roi_image_path)

    # Imprimir los dígitos extraídos
    print("Dígitos extraídos de la imagen:", extracted_digits)
