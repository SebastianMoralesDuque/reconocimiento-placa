import cv2
import pytesseract
import mysql.connector
import os
from fuzzywuzzy import fuzz
os.environ['TESSDATA_PREFIX'] = 'tesseract-ocr/tessdata'

# Conectar a la base de datos de MariaDB
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="placas"
)
mycursor = mydb.cursor()

# Cargar la imagen de la placa
url='placas/qw.jpg'
img = cv2.imread(url)

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro Gaussiano para suavizar la imagen
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# Aplicar la binarización adaptativa para resaltar los bordes de la placa
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)

# Encontrar los contornos de la placa
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def get_similarity(string1, string2):
    similarity_ratio = fuzz.ratio(string1.lower(), string2.lower()) / 100
    return similarity_ratio

# Seleccionar el contorno más grande (que debería ser la placa)
if contours:
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Dibujar un rectángulo alrededor de la placa
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Recortar la imagen de la placa
    plate_img = gray[y:y+h, x:x+w]

    # Obtener el texto de la placa utilizando Tesseract OCR
    text = pytesseract.image_to_string(plate_img, lang='spa', config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_font_size=12 --oem 1').strip()
    # Obtener el resultado correcto del usuario y calcular la precisión
    resultado_correcto = input("Ingrese el resultado correcto:")

    precision = get_similarity(text,resultado_correcto);

    # Guardar los datos en la base de datos
    sql = "INSERT INTO placas (imagen, texto_placa, resultado_correcto, prec) VALUES (%s, %s, %s, %s)"
    val = (url, text, resultado_correcto.upper(), precision)
    mycursor.execute(sql, val)
    mydb.commit()

    print(mycursor.rowcount, "registro insertado.")
else:
    print("No se encontró una placa en la imagen.")
