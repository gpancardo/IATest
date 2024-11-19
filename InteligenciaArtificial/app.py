import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from utils import preprocess_image
import numpy as np

app = Flask(__name__)

# Configuración de la carpeta de carga
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
