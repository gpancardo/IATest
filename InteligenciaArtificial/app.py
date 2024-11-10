import os
from flask import Flask, request, render_template
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import cv2

# Inicializar la aplicación Flask
app = Flask(__name__)

# Configuración de la carpeta de carga
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# 1. Cargar y preprocesar los datos (MNIST)
def load_data():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

# 2. Definir la arquitectura de la red neuronal
def create_model(num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Entrenar y guardar el modelo
def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    model.save('models/character_recognition_model.h5')
    print("Modelo guardado en 'models/character_recognition_model.h5'.")

# 4. Preprocesamiento de la imagen subida para hacer predicciones
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255
    img = img.reshape(1, 28, 28, 1)
    return img

# 5. Ruta para cargar la imagen y predecir el número
@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part in the request", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file:
            # Guardar la imagen subida
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Cargar el modelo guardado
            model = load_model('models/character_recognition_model.h5')
            
            # Preprocesar la imagen y hacer la predicción
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)

            return render_template("upload.html", prediction=predicted_class, img_path=file_path)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
