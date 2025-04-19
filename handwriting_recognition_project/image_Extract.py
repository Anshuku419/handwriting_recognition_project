import cv2
import pytesseract
import numpy as np

# Set the correct path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image_path = "debug_processed_image.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding for better OCR
processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Perform OCR
extracted_text = pytesseract.image_to_string(processed, lang="eng", config="--oem 3 --psm 6")

print("Extracted Text:\n", extracted_text)
