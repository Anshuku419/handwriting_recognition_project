import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Update Tesseract Path (Windows Users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """
    Preprocessing Steps to Improve OCR:
    - Convert to grayscale
    - Apply thresholding
    - Remove noise
    - Enhance text using morphological operations
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Adaptive Thresholding to enhance contrast
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)

    # Denoise and enhance text
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save processed image for debugging
    cv2.imwrite("debug_processed_image.png", processed)

    return processed

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    # Convert image to OpenCV format
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess the image
    processed_img = preprocess_image(img)

    # Extract text with Hindi + English support
    extracted_text = pytesseract.image_to_string(
        processed_img,
        lang="eng+hin",
        config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
    )

    return jsonify({"prediction": extracted_text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
