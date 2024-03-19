from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_explain.core.grad_cam import GradCAM


app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_brain_tumor_model():
    model_path = 'models/brain_tumor_model.h5'
    return load_model(model_path)

def load_alzheimer_model():
    model_path = 'models/alzheimer_model.h5'
    return load_model(model_path)

brain_tumor_model_loaded = load_brain_tumor_model()
alzheimer_model_loaded = load_alzheimer_model()

def preprocess_brain_tumor_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

def preprocess_alzheimer_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (176, 176))
    img = img / 255.0
    return img


def grad_cam(image_path, model, layer_name='conv2d_10'):
    img = preprocess_alzheimer_image(image_path)
    img_batch = np.expand_dims(img, axis=0)

    explainer = GradCAM()
    # Assuming 3 is the index for the Alzheimer's class
    class_index = 3

    grid = explainer.explain((img_batch, None), model, class_index, layer_name)

    explanation_path = 'static/grad_cam.png'
    explainer.save(grid, ".", explanation_path)  # Add the output_dir argument

    return explanation_path


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

      

        img = preprocess_brain_tumor_image(file_path)
        img_batch = np.expand_dims(img, axis=0)
        brain_tumor_prediction = brain_tumor_model_loaded.predict(img_batch)

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

        img = preprocess_alzheimer_image(file_path)
        img_batch = np.expand_dims(img, axis=0)
        alzheimer_prediction = alzheimer_model_loaded.predict(img_batch)

        alzheimer_result = np.argmax(alzheimer_prediction[0])
        alzheimer_classes = ['MildDemented', 'VeryMildDemented', 'NonDemented', 'ModerateDemented']
        alzheimer_result_label = alzheimer_classes[alzheimer_result]

        explanation_path = grad_cam(file_path, alzheimer_model_loaded, layer_name='conv2d_10')

        return render_template('result_alzheimer.html', filename=filename,r=alzheimer_result, alzheimer_result=alzheimer_result_label, grad_cam_path=explanation_path)

    else:
        return render_template('index.html', error='Invalid file format')

if __name__ == '__main__':
    app.run(debug=True)

