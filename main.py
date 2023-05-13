import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# definir el modelo
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# cargar los datos de entrenamiento
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalizar los datos
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# entrenar el modelo
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=2, validation_split=0.1)

# evaluar el modelo
model.evaluate(x_test, y_test)

# guardar el modelo
model.save("modelo_entrenado.h5")

# Cargar el modelo entrenado de TensorFlow
model = tf.keras.models.load_model('modelo_entrenado.h5')

# Leer la imagen
img = cv2.imread('placa.jpg')

# Convertir la imagen a escala de grises y aplicar un filtro gaussiano para reducir el ruido
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplicar la transformación de umbral adaptativo para convertir la imagen en blanco y negro
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Encontrar los contornos de la imagen
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Inicializar la lista de caracteres reconocidos
recognized_chars = []

# Iterar sobre cada contorno
for contour in contours:
    # Obtener el rectángulo que rodea el contorno
    (x, y, w, h) = cv2.boundingRect(contour)

    # Ignorar los contornos que son muy pequeños o muy grandes
    if w < 5 or h < 5 or w > 100 or h > 100:
        continue

    # Recortar la región de interés (ROI) de la imagen
    roi = thresh[y:y+h, x:x+w]

    # Cambiar el tamaño de la ROI a 28x28 (el tamaño de entrada del modelo)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalizar los valores de píxeles de la ROI a un rango de 0 a 1
    roi = roi.astype("float32") / 255.0

    # Aplanar la ROI en un vector unidimensional
    roi = np.reshape(roi, (1, 28, 28, 1))

    # Hacer una predicción con el modelo
    pred = model.predict(roi)

    # Obtener el índice del valor máximo de predicción (es decir, el dígito reconocido)
    index = np.argmax(pred)

    # Convertir el índice a un carácter ASCII (0-9)
    char = chr(index + 48)

    # Añadir el carácter reconocido a la lista de caracteres reconocidos
    recognized_chars.append(char)

# Unir los caracteres reconocidos en un string
recognized_text = "".join(recognized_chars)

# Mostrar el resultado
print(recognized_text)
