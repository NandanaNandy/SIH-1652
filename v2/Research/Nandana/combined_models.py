import os
from pathlib import Path
import torch  
from transformers import pipeline
import mimetypes
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import langid
import fasttext
import logging
from datetime import datetime
import warnings
warnings.simplefilter("ignore", FutureWarning)

class DocumentProcessor:
    def __init__(self, model_name="papluca/xlm-roberta-base-language-detection"):
        self.logger = self.setup_logging()
        self.lang_detector = self.load_language_model(model_name)
        self.fasttext_model = self.load_fasttext_model()
        self.langid_model = self.load_langid_model()

    def setup_logging(self):
        log_file = f"processing_{datetime.now().strftime('%Y-%m-%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        return logging.getLogger(__name__)

    def load_language_model(self, model_name):
        self.logger.info("Loading XLM-RoBERTa language detection model...")
        return pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

    def load_fasttext_model(self):
        self.logger.info("Loading fastText model for language detection...")
        return fasttext.load_model("lid.176.bin")

    def load_langid_model(self):
        self.logger.info("Loading langid model...")
        langid.set_languages(['en', 'ta', 'hi', 'ml'])  # Pre-set supported languages

    def detect_file_type(self, file_path):
        file_type, _ = mimetypes.guess_type(file_path)
        return file_type or "Unknown"

    def extract_text(self, file_path, file_type):
        try:
            if file_type.startswith("image"):
                return pytesseract.image_to_string(Image.open(file_path))
            elif file_type == "application/pdf":
                reader = PdfReader(file_path)
                return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif file_type == "text/plain":
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error extracting text from file: {e}")
            return None

    def analyze_languages(self, text):
        # Using XLM-RoBERTa model
        xlm_roberta_results = self.lang_detector(text)
        # Using fastText model
        fasttext_results = self.fasttext_model.predict(text, k=3)  # Get top 3 languages
        # Using langid model
        langid_result = langid.classify(text)

        # Combine all results
        combined_results = {}

        # XLM-RoBERTa Results
        for result in xlm_roberta_results:
            lang = result['label']
            score = result['score']
            combined_results[lang] = combined_results.get(lang, 0) + score

        # fastText Results
        for lang, score in zip(fasttext_results[0], fasttext_results[1]):
            combined_results[lang] = combined_results.get(lang, 0) + score

        # langid Result
        langid_lang, langid_score = langid_result
        combined_results[langid_lang] = combined_results.get(langid_lang, 0) + langid_score

        # Normalize and filter out low scores
        total_score = sum(combined_results.values())
        lang_percentages = {
            lang: round((score / total_score) * 100, 2)
            for lang, score in combined_results.items() if (score / total_score) * 100 > 5.0
        }

        return lang_percentages

    def process_file(self, file_path):
        file_type = self.detect_file_type(file_path)
        text = self.extract_text(file_path, file_type)
        if text:
            lang_results = self.analyze_languages(text)
            return file_type, text[:200], lang_results  # Limit text snippet to 200 characters
        else:
            return file_type, "No text could be extracted.", {}

def main():
    processor = DocumentProcessor()

    while True:
        file_path = input("Enter the file path (or type 'exit' to quit): ").strip()
        if file_path.lower() == 'exit':
            print("Exiting program.")
            break

        if not Path(file_path).is_file():
            print(f"Invalid file path: {file_path}")
            continue

        file_type, text_snippet, lang_results = processor.process_file(file_path)
        print("\n--- File Analysis ---")
        print(f"File Type: {file_type}")
        print(f"Extracted Text (Snippet): {text_snippet}")
        print("Language Distribution:")
        for lang, percentage in lang_results.items():
            print(f"  {lang}: {percentage}%")
        print("---------------------\n")

if __name__ == "__main__":
    main()

