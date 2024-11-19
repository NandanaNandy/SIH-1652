import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import os
import threading

# Lock for thread-safe file writing
file_lock = threading.Lock()

def extract_text_from_image(image_path: str) -> str:
    """Extract text from a single image file."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error processing {image_path}: {e}"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages of a PDF file."""
    try:
        pages = convert_from_path(pdf_path)
        all_text = []
        for page_number, page in enumerate(pages, start=1):
            text = pytesseract.image_to_string(page)
            all_text.append(f"Page {page_number}:\n{text.strip()}")
        return "\n".join(all_text)
    except Exception as e:
        return f"Error processing {pdf_path}: {e}"

def process_file(file_path: str, output_file: str):
    """Process a file and write structured output line by line."""
    if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        text = extract_text_from_image(file_path)
    elif file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        text = f"Unsupported file format: {file_path}"
    
    # Write the result to the output file in a thread-safe manner
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"File: {file_path}\n{text}\n{'-'*80}\n")

def extract_text_multithreaded(file_paths, output_file: str, num_threads: int = 4):
    """Extract text from multiple files using threads and write line by line."""
    # Clear the output file before starting
    open(output_file, "w").close()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_file, file_paths, [output_file] * len(file_paths))

# Example usage
if __name__ == "__main__":
    # Replace with actual file paths
    uploaded_files = [
        "F:\Softcopy _official\IMG-20221112-WA0001.jpg",
        "F:\Softcopy _official\Screenshot_2024_0115_094707.jpg",
        "F:\Softcopy _official\voter id.pdf",
    ]

    # Ensure files exist
    uploaded_files = [file for file in uploaded_files if os.path.exists(file)]

    output_file = "extracted_text.txt"

    if not uploaded_files:
        print("No valid files found!")
    else:
        # Extract text using threads
        extract_text_multithreaded(uploaded_files, output_file, num_threads=4)
        print(f"Text extracted and written to {output_file}")
