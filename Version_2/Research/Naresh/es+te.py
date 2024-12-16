import easyocr
import pytesseract
from pdf2image import convert_from_path
import json
import re
import cv2
import numpy as np

# Initialize EasyOCR and Tesseract
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Path to Poppler's `bin` directory
poppler_path = r"C:\Users\nares\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

def extract_text_from_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform OCR using EasyOCR
    results = ocr_reader.readtext(gray_image, detail=0)
    return ' '.join(results)

def process_pdf(file_path):
    # Convert PDF pages to images, specifying the Poppler path
    images = convert_from_path(file_path, poppler_path=poppler_path)
    extracted_text = ""

    for i, image in enumerate(images):
        # Convert PIL image to OpenCV format
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Extract text from each page image
        text = extract_text_from_image(open_cv_image)
        extracted_text += text + "\n"
    
    return extracted_text

def structure_text_to_json(extracted_text):
    # Structure text into sections for JSON
    data = {}
    
    # Extract sections using regex (simple example for demonstration)
    data['name'] = re.search(r"Name:\s*(.*)", extracted_text)
    data['location'] = re.search(r"Location:\s*(.*)", extracted_text)
    data['phone'] = re.search(r"Phone:\s*(\d{10})", extracted_text)
    data['email'] = re.search(r"Email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", extracted_text)
    
    # Extract skills and experiences, based on known keywords
    skills = re.search(r"SKILLS\s+(.*?)\s+EXPERIENCES", extracted_text, re.S)
    experiences = re.search(r"EXPERIENCES\s+(.*?)\s+EDUCATION", extracted_text, re.S)
    education = re.search(r"EDUCATION\s+(.*?)\s+ACHIEVEMENTS", extracted_text, re.S)
    achievements = re.search(r"ACHIEVEMENTS\s+(.*?)\s+COURSES", extracted_text, re.S)
    projects = re.search(r"PROJECTS\s+(.*?)$", extracted_text, re.S)
    
    # Populate structured JSON
    data['skills'] = skills.group(1).strip().split() if skills else []
    data['experiences'] = experiences.group(1).strip() if experiences else ""
    data['education'] = education.group(1).strip() if education else ""
    data['achievements'] = achievements.group(1).strip() if achievements else ""
    data['projects'] = projects.group(1).strip() if projects else ""
    
    return data

def main(file_path):
    # Process the PDF and extract text
    extracted_text = process_pdf(file_path)
    
    # Structure extracted text into JSON
    structured_data = structure_text_to_json(extracted_text)
    
    # Output JSON
    print(json.dumps(structured_data, indent=4))

# Run the main function with the PDF path
file_path = "AnandaKrishnan.pdf"
main(file_path)
