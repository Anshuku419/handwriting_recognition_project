import cv2
import pytesseract
import numpy as np

# Set the Tesseract path manually (Modify this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image_path = "debug_processed_image.png"  # Update with your image path
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to remove noise
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Apply morphological operations to improve clarity
kernel = np.ones((2,2), np.uint8)
processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Save processed image for debugging (Optional)
cv2.imwrite("debug_processed_image1.png", processed)

# Perform OCR
extracted_text = pytesseract.image_to_string(processed, lang="eng", config="--oem 3 --psm 6")

# Print the extracted text
print("Extracted Text:\n", extracted_text)
