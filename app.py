from flask import Flask, render_template, request
import os
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow as tf
import cv2
import re
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

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    return img_array

def proses_prediksi_ke_label(pred):
    class_idx = np.argmax(pred[0])
    labels = ['Glioma', 'Meningioma', 'Pituitary', 'Normal']
    return labels[class_idx]

app = Flask(__name__)

model = YOLO('model/best.pt')
efficientnet_model = load_model('model/effnet.keras')

def get_gradcam_heatmap(model, image, class_index, last_conv_layer_name='top_activation'):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap  

def save_gradcam(img_array, heatmap, save_path, alpha=0.4):
    img = np.uint8(255 * img_array[0])  
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(save_path, superimposed_img)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    gradcam_path = None
    prediction = None
    explanation = None
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join('static', image.filename)
        image.save(image_path)
        model_choice = request.form['model_choice']
        img_array = preprocess_image(image_path) 
    
        if model_choice == 'resnet':
            pred = resnet_model.predict(img_array)
            gradcam_path = image_path  
        elif model_choice == 'vgg':
            pred = vgg_model.predict(img_array)
            gradcam_path = image_path
        elif model_choice == 'densenet':
            pred = densenet_model.predict(img_array)
            gradcam_path = image_path
        elif model_choice == 'efficientnet':
            pred = efficientnet_model.predict(img_array)
            pred_label = np.argmax(pred[0])
            heatmap = get_gradcam_heatmap(
                efficientnet_model, img_array[0], class_index=pred_label, last_conv_layer_name='top_conv'
            )
            gradcam_filename = f'gradcam_{image.filename}'
            gradcam_path = gradcam_filename
            save_gradcam(img_array, heatmap, os.path.join('static', gradcam_filename))
        else:
            pred = None
            gradcam_path = None

        prediction = proses_prediksi_ke_label(pred)
        explanation = "Penjelasan GradCAM oleh LLM" 

    return render_template('classify.html', gradcam_path=gradcam_path, prediction=prediction, explanation=explanation)

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
            "Jelaskan hasil segmentasi tumor otak pada gambar MRI ini secara singkat dan jelasz. "
        )
        segmentation_info = get_gemini_explanation(prompt)
        segmentation_info = bersihkan_teks(segmentation_info) 

    return render_template('segment.html', segmented_path=segmented_path, segmentation_info=segmentation_info)

if __name__ == '__main__':
    app.run(debug=True)