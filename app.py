from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import logging
import os

# Set the environment variable `TF_ENABLE_ONEDNN_OPTS` to `0`
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Load the model
model_final = tf.keras.models.load_model('model_final_adam.h5')

# Class mapping
class_mapping = {
    0: 'Deoni',
    1: 'Gir',
    2: 'Kankrej',
    3: 'Rathi',
    4: 'Sahiwaal',
    5: 'Sindhi',
    6: 'Tharparkar'
}

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif','webp'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Setup logging
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    image_filename = None
    confidence = None  # Initialize confidence

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")

        file = request.files['file']
        logging.info("File received: %s", file.filename)

        if file.filename == '':
            return render_template('index.html', prediction="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

                prediction = model_final.predict(img_array)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
###################################### CONFIDENCE CODE ##########################################################
                # Confidence code 
                confidence = np.max(prediction[0]) * 100
                confidence = round(confidence, 2)  # round off to two decimal places
                print("Confidence", confidence,"%")
################################################################################################
                predicted_class_name = class_mapping.get(predicted_class_index, "Unknown")
                image_filename = filename

                # Optional: Delete the uploaded file after processing
                # os.remove(file_path)

                return render_template('index.html', prediction=predicted_class_name, image=image_filename, confidence=confidence)

            except Exception as e:
                logging.error("Error during processing: %s", str(e))
                return render_template('index.html', prediction=f"Error during processing: {str(e)}")

    return render_template('index.html', prediction=prediction, image=image_filename, confidence=confidence)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)