import cv2
import numpy as np
import pyopencl as cl
from PIL import Image
import pytesseract
import os

# Set Tesseract command path (Update if different on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define language codes for OCR (customize based on requirement)
langs = "eng+tam"  # Example: English + Tamil

# Load the image (replace 'image.jpg' with your image path)
image_path = 'unstop_Flipkart.jpg'  # Update the image path if needed

# Verify if the image exists
if not os.path.exists(image_path):
    print(f"Error: The image file '{image_path}' does not exist.")
    exit(1)

# Load the image using OpenCV
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Failed to load image from '{image_path}'.")
    exit(1)
else:
    print(f"Image loaded successfully from '{image_path}'.")

# Preprocessing with OpenCL
# OpenCL setup: create context and queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Convert image to grayscale using OpenCV
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Contrast adjustment
alpha = 1.5  # Contrast control, >1 to increase contrast
beta = 10    # Brightness control, slightly adjust as needed
adjusted = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(adjusted, (3, 3), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 4)

# Sharpen the image
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
sharpened = cv2.filter2D(thresh, -1, sharpen_kernel)

# Perform morphological operations for further noise reduction
kernel = np.ones((1, 1), np.uint8)
processed_image = cv2.erode(sharpened, kernel, iterations=1)
processed_image = cv2.dilate(processed_image, kernel, iterations=1)

# Convert processed image to PIL format for Tesseract OCR
pil_img = Image.fromarray(processed_image)

# Tesseract OCR with optimized configurations
custom_config = r'--oem 3 --psm 6'  # Adjust to your needs
text = pytesseract.image_to_string(pil_img, lang=langs, config=custom_config)

print("Extracted Text:\n", text)

# Display the result using OpenCV (optional)
cv2.imshow("Processed Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
