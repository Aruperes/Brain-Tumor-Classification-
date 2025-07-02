from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    gradcam_path = None
    prediction = None
    explanation = None
    if request.method == 'POST':
        # Simulasi proses prediksi dan gradcam
        # Ganti dengan pemanggilan model & gradcam Anda
        image = request.files['image']
        image.save(os.path.join('static', image.filename))
        gradcam_path = f'static/{image.filename}'  # Dummy, ganti dengan hasil gradcam
        prediction = "Glioma"  # Dummy, ganti dengan hasil prediksi model
        explanation = "Penjelasan GradCAM oleh LLM."  # Dummy, ganti dengan hasil LLM
    return render_template('classify.html', gradcam_path=gradcam_path, prediction=prediction, explanation=explanation)

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    segmented_path = None
    segmentation_info = None
    if request.method == 'POST':
        image = request.files['image']
        image.save(os.path.join('static', image.filename))
        segmented_path = image.filename  # Dummy, ganti dengan hasil segmentasi
        segmentation_info = "Penjelasan segmentasi oleh model."  # Dummy
    return render_template('segment.html', segmented_path=segmented_path, segmentation_info=segmentation_info)


if __name__ == '__main__':
    app.run(debug=True)