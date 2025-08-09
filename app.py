from flask import Flask, render_template, request
import os
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_explanation(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

def bersihkan_teks(teks):
    teks = re.sub(r'[*_`#>\\-]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

LABELS = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
IMAGE_SIZE = 224

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def load_selected_model(model_name):
    if model_name == 'efficientnet':
        model = load_model('model/effnet.h5')
        last_conv = 'top_conv'
    elif model_name == 'resnet':
        model = load_model('model/resnet.keras')
        last_conv = 'layer akhir'
    elif model_name == 'vgg':
        model = load_model('model/vgg.keras')
        last_conv = 'layer akhir'
    elif model_name == 'densenet':
        model = load_model('model/densenet.keras')
        last_conv = 'layer akhir'
    else:
        raise ValueError("Unknown model")
    return model, last_conv
    

app = Flask(__name__)

model = YOLO('model/yolo.pt')

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    prediction = None
    gradcam_filename = None  
    selected_model = None
    if request.method == 'POST':
        selected_model = request.form['model']
        img_file = request.files['image']
        img_path = os.path.join('static', img_file.filename)
        img_file.save(img_path)

        # Preprocess image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(img, axis=0)

        # Load model
        model, last_conv = load_selected_model(selected_model)
        preds = model.predict(img_array)
        pred_class = LABELS[np.argmax(preds)]
        prediction = pred_class

        # Grad-CAM
        gradcam_filename = f'gradcam_{img_file.filename}'
        gradcam_path = os.path.join('static', gradcam_filename)
        heatmap = get_gradcam_heatmap(model, img_array, last_conv)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(gradcam_path, superimposed_img)

    return render_template(
        'classify.html',
        prediction=prediction,
        gradcam_path=gradcam_filename, 
        selected_model=selected_model
    )

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    segmented_path = None
    segmentation_info = None
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join('static', image.filename)
        image.save(image_path)
        results = model(image_path)
        result_img_path = f'static/hasil_{image.filename}'
        results[0].save(filename=result_img_path)
        segmented_path = f'hasil_{image.filename}'
        prompt = (
            "Jelaskan hasil segmentasi tumor otak pada gambar MRI ini secara singkat, jelas, dan langsung ke inti. "
        )
        segmentation_info = get_gemini_explanation(prompt)
        segmentation_info = bersihkan_teks(segmentation_info) 

    return render_template('segment.html', segmented_path=segmented_path, segmentation_info=segmentation_info)

if __name__ == '__main__':
    app.run(debug=True)