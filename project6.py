import cv2
import pytesseract
import os
from tkinter import *

# Configuración de Tesseract

def reconocer_digitos(imagen):
    """Reconoce dígitos de una imagen."""
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización adaptativa
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Extraer dígitos usando Tesseract
    config = "--psm 10 -c tessedit_char_whitelist=0123456789"  # Modo caracter individual, lista blanca de dígitos
    texto = pytesseract.image_to_string(umbral, config=config)

    # Extraer solo los dígitos
    digitos = [int(d) for d in texto if d.isdigit()]
    return digitos

def mostrar_imagen(imagen, ventana):
    """Muestra una imagen en la ventana."""
    alto, ancho, _ = imagen.shape
    imagen_redimensionada = cv2.resize(imagen, (ancho // 2, alto // 2))
    imagen_tk = PhotoImage(image=imagen_redimensionada)
    etiqueta_imagen = Label(ventana, image=imagen_tk)
    etiqueta_imagen.pack()

def procesar_imagen(ruta_imagen, ventana):
    """Procesa una imagen y muestra los resultados en la ventana."""
    imagen = cv2.imread(ruta_imagen)
    digitos = reconocer_digitos(imagen)

    # Mostrar imagen
    mostrar_imagen(imagen, ventana)

    # Mostrar mensaje de dígitos
    if digitos:
        texto_digitos = "Dígitos: " + "".join(str(d) for d in digitos)
        etiqueta_digitos = Label(ventana, text=texto_digitos)
        etiqueta_digitos.pack()
    else:
        etiqueta_digitos = Label(ventana, text="Imagen no clara o visible")
        etiqueta_digitos.pack()

def main():
    # Crear ventana principal
    ventana = Tk()
    ventana.geometry("600x400")
    ventana.title("Reconocimiento de Dígitos en Medidores de Luz")

    # Bucle para procesar imágenes
    for archivo in os.listdir("imagenes"):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta_imagen = os.path.join("imagenes", archivo)
            procesar_imagen(ruta_imagen, ventana)

    # Iniciar bucle principal de la ventana
    ventana.mainloop()

if __name__ == "__main__":
    main()
