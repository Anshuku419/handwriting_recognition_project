import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, render_template


app = Flask(__name__, template_folder="templates", static_folder="static")


# Update Tesseract path (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_stylish_image(image):
    """
    Preprocess stylish images:
    - Convert to grayscale
    - Enhance contrast
    - Apply adaptive thresholding
    - Sharpen the text
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    enhanced = cv2.convertScaleAbs(gray, alpha=2, beta=10)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Morphological operations to clean noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    # Convert image to OpenCV format
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess the image
    processed_img = preprocess_stylish_image(img)

    # Extract text in Hindi, English, and digits
    extracted_text = pytesseract.image_to_string(
        processed_img,
        lang="eng+hin",  # English and Hindi language support
        config="--oem 3 --psm 6"
    )
    return jsonify({"prediction": extracted_text.strip()})
if __name__ == '__main__':
    app.run(debug=True)