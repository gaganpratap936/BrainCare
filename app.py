from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_brain_tumor_model():
    # Load pre-trained brain tumor detection model
    model_path = 'models/brain_tumor_model.h5'
    return load_model(model_path)  # Provide the correct path to your trained model file

def load_alzheimer_model():
    # Load pre-trained Alzheimer's disease detection model
    model_path = 'models/alzheimer_model.h5'
    return load_model(model_path)  

brain_tumor_model_loaded = load_brain_tumor_model()
alzheimer_model_loaded = load_alzheimer_model()

def preprocess_brain_tumor_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize the image to the expected shape
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    # Perform any other preprocessing steps if needed
    return img

def preprocess_alzheimer_image(image_path):
    # Load and preprocess the input image for Alzheimer's detection
    img = cv2.imread(image_path)
    img = cv2.resize(img, (176, 176))  # Resize the image to the expected shape
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    # Perform any other preprocessing steps if needed
    return img

def crop_and_preprocess_image(img):
    # Implement your custom function for brain tumor image preprocessing (e.g., cropping)
    # Ensure the output shape matches the input shape expected by the model
    return img

def preprocess_image(img):
    # Implement your custom function for Alzheimer image preprocessing
    # Ensure the output shape matches the input shape expected by the model
    return img

def predict_brain_tumor(image_path, model):
    # Make prediction for brain tumor using the pre-trained model
    img = preprocess_brain_tumor_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

def predict_alzheimer(image_path, model):
    # Make prediction for Alzheimer's using the pre-trained model
    img = preprocess_alzheimer_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

@app.route('/result_bt', methods=['POST'])
def upload_brain_tumor_file():
    if 'file' not in request.files:
        return render_template('braintumor.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('braintumor.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make predictions for brain tumor
        img = preprocess_brain_tumor_image(file_path)
        img_batch = np.expand_dims(img, axis=0)
        brain_tumor_prediction = brain_tumor_model_loaded.predict(img_batch)

        # Perform further analysis on brain tumor predictions as needed

        if brain_tumor_prediction < 0.7:
            brain_tumor_result = 0
        else:
            brain_tumor_result = 1

        return render_template('result_bt.html', filename=filename, r=brain_tumor_result)

    else:
        return render_template('index.html', error='Invalid file format')

@app.route('/result_alzheimer', methods=['POST'])
def upload_alzheimer_file():
    if 'file' not in request.files:
        return render_template('alzheimer.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('alzheimer.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make predictions for Alzheimer's
        img = preprocess_alzheimer_image(file_path)
        img_batch = np.expand_dims(img, axis=0)
        alzheimer_prediction = alzheimer_model_loaded.predict(img_batch)

        # Perform further analysis on Alzheimer's predictions as needed

        alzheimer_result = np.argmax(alzheimer_prediction[0])
        alzheimer_classes = ['MildDemented', 'VeryMildDemented', 'NonDemented', 'ModerateDemented']
        alzheimer_result_label = alzheimer_classes[alzheimer_result]

        # Check if the predicted class is "ModerateDemented"
        #if alzheimer_result == alzheimer_classes.index('ModerateDemented'):
           # alzheimer_result_label = alzheimer_classes[alzheimer_result]
        return render_template('result_alzheimer.html', filename=filename, alzheimer_result=alzheimer_result_label)
        #else:
            # Redirect to another route if the predicted class is not "ModerateDemented"
            #return redirect(url_for('error_route', filename=filename))

    else:
        return render_template('index.html', error='Invalid file format')

if __name__ == '__main__':
    app.run(debug=True)