from flask import Flask, render_template, request
import os
from ultralytics import YOLO
import google.generativeai as genai
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

app = Flask(__name__)

model = YOLO('model/best.pt')

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

        # Preprocess gambar sesuai kebutuhan model
        # Misal: img_array = preprocess_image(image_path)

        # Pilih model sesuai pilihan user
        if model_choice == 'resnet':
            pred = resnet_model.predict(img_array)
        elif model_choice == 'vgg':
            pred = vgg_model.predict(img_array)
        elif model_choice == 'densenet':
            pred = densenet_model.predict(img_array)
        elif model_choice == 'efficientnet':
            pred = efficientnet_model.predict(img_array)
        else:
            pred = None

        # Proses hasil prediksi ke label
        prediction = proses_prediksi_ke_label(pred)
        gradcam_path = image_path  # Ganti dengan hasil gradcam jika ada
        explanation = "Penjelasan GradCAM oleh LLM."  # Ganti dengan hasil LLM jika ada

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
            "Jelaskan hasil segmentasi tumor otak pada gambar MRI ini secara singkat, jelas, dan langsung ke inti. "
        )
        segmentation_info = get_gemini_explanation(prompt)
        segmentation_info = bersihkan_teks(segmentation_info) 

    return render_template('segment.html', segmented_path=segmented_path, segmentation_info=segmentation_info)

if __name__ == '__main__':
    app.run(debug=True)