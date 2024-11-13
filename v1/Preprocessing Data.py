!pip install pytesseract
!sudo apt-get install tesseract-ocr
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os

# Check if the input is a PDF or an image
def process_file(file_path):
    if file_path.endswith('.pdf'):
        # If the file is a PDF, convert it to images
        images = convert_from_path(file_path)

        for i, image in enumerate(images):
            # Save pages as images in the current directory
            image.save(f'page_{i}.jpg', 'JPEG')

            # Process each page image
            process_image(f'page_{i}.jpg')

    elif file_path.endswith(('.jpg', '.jpeg', '.png')):
        # If the file is an image, process it directly
        process_image(file_path)
    else:
        print("Unsupported file format. Please provide a PDF or an image file.")

# Function to process an image (load, grayscale, threshold, OCR)
def process_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        # Step 1: Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply thresholding
        _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Step 3: OCR the thresholded image using Tesseract
        text = pytesseract.image_to_string(thresholded_image, lang='eng')

        # Print the extracted text
        print(f"Extracted Text from {image_path}:\n", text)

        # (Optional) Display the grayscale and thresholded images
        from google.colab.patches import cv2_imshow
        cv2_imshow(gray_image)  # Display the grayscale image
        cv2_imshow(thresholded_image)  # Display the thresholded image

        # Save the processed images if needed
        cv2.imwrite(f'gray_image_{os.path.basename(image_path)}', gray_image)
        cv2.imwrite(f'thresholded_image_{os.path.basename(image_path)}', thresholded_image)

# Path to the file (PDF or image)
file_path = r"/content/Comm.pdf"  # Change this to your PDF or image file path

# Process the file
process_file(file_path)
