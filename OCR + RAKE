import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from rake_nltk import Rake
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return ""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresholded_image, lang='eng')
        return text
    except Exception as e:
        print(f"Error processing image: {e}")
        return ""

def process_pdf(file_path):
    try:
        images = convert_from_path(file_path)
        extracted_text = ""
        for i, image in enumerate(images):
            image_path = f'page_{i}.jpg'
            image.save(image_path, 'JPEG')
            extracted_text += extract_text_from_image(image_path) + "\n"
        return extracted_text
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return ""

def preprocess_text(text):
    # Remove special characters and extra whitespace
    cleaned_text = re.sub(r'[^\w\s,.]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def extract_name_and_dob(text):
    name_match = re.search(r"(?i)name:\s*(.*)", text)
    dob_match = re.search(r"(?i)date of birth\s*\(dd-mm-yyyy\):\s*(\d{2}-\d{2}-\d{4})", text)
    extracted_name = name_match.group(1).strip() if name_match else None
    extracted_dob = dob_match.group(1).strip() if dob_match else None
    return extracted_name, extracted_dob

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def compare_extracted_with_user_data(extracted_text, user_data):
    comparison_result = {}
    extracted_name, extracted_dob = extract_name_and_dob(extracted_text)

    # Print extracted and expected values for debugging
    print("Extracted Name:", extracted_name)
    print("Expected Name:", user_data.get('name'))
    print("Extracted Date of Birth:", extracted_dob)
    print("Expected Date of Birth:", user_data.get('dob'))

    # Normalize and compare name with a fallback
    comparison_result['Name'] = (
        extracted_name.lower() == user_data.get('name').lower() if extracted_name else False
    )

    # Compare date of birth
    comparison_result['Date of Birth'] = (
        extracted_dob == user_data.get('dob') if extracted_dob else False
    )

    return comparison_result


def main(file_path, user_data):
    if file_path.endswith('.pdf'):
        extracted_text = process_pdf(file_path)
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        extracted_text = extract_text_from_image(file_path)
    else:
        print("Unsupported file format. Please provide a PDF or image file.")
        return

    cleaned_text = preprocess_text(extracted_text)
    print("Cleaned Extracted Text:\n", cleaned_text)

    keywords = extract_keywords(cleaned_text)
    print("\nExtracted Keywords:\n", keywords)

    entities = extract_entities(cleaned_text)
    print("\nExtracted Entities:\n", entities)

    comparison_result = compare_extracted_with_user_data(cleaned_text, user_data)
    print("\nComparison Result:\n", comparison_result)

# Example user-provided data
user_data = {
    'name': 'Nareshkanna S',
    'dob': '15-04-2005',
}

file_path = r"/content/Comm.pdf"  # Replace with the actual file path

main(file_path, user_data)
