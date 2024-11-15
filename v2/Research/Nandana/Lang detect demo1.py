import os
from pathlib import Path
from transformers import pipeline
import torch
import logging
import shutil
from datetime import datetime

class DocumentPaths:
    def __init__(self, base_dir: str = None):
        self.logger = logging.getLogger(__name__)
        
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        self.paths = {
            'uploads': self.base_dir / 'uploads',
            'temp': self.base_dir / 'temp',
            'output': self.base_dir / 'output',
            'models': self.base_dir / 'models',
            'logs': self.base_dir / 'logs',
        }
        
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Create necessary directories."""
        try:
            for path in self.paths.values():
                path.mkdir(parents=True, exist_ok=True)
                if os.name != 'nt':  # For Unix-like systems
                    os.chmod(path, 0o755)
            self.logger.info("Directories initialized.")
        except Exception as e:
            self.logger.error(f"Directory setup failed: {e}")
            raise

    def setup_logging(self):
        """Configure logging."""
        log_file = self.paths['logs'] / f'processing_{datetime.now().strftime("%Y-%m-%d")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

def initialize_document_processor(base_dir: str = None, model_name: str = "papluca/xlm-roberta-base-language-detection"):
    """Initialize the document processor."""
    try:
        doc_paths = DocumentPaths(base_dir)
        lang_detector = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            cache_dir=str(doc_paths.paths['models'])
        )
        return doc_paths, lang_detector
    except Exception as e:
        logging.error(f"Failed to initialize document processor: {e}")
        raise

def detect_file_type(file_path):
    """Determine if a file is text or binary."""
    try:
        with open(file_path, 'rb') as file:
            content = file.read(1024)  # Read a small portion
            if b'\0' in content:
                return "Binary"
            return "Text"
    except Exception as e:
        logging.error(f"Failed to detect file type: {e}")
        return "Unknown"

def process_file(file_path, lang_detector):
    """Process a single file."""
    try:
        file_type = detect_file_type(file_path)
        if file_type == "Text":
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            language = lang_detector(content[:512])[0]['label']  # Detect language using the model
            return content[:200], file_type, language  # Return snippet for brevity
        else:
            return "Binary data (not processed)", file_type, "N/A"
    except Exception as e:
        logging.error(f"Failed to process file: {e}")
        return None, "Error", "Error"

if __name__ == "__main__":
    doc_paths, lang_detector = initialize_document_processor()

    print("Enter file paths one by one (type 'exit' to quit):")
    while True:
        file_path = input("File path: ").strip()
        if file_path.lower() == 'exit':
            print("Exiting program.")
            break
        if not Path(file_path).is_file():
            print(f"Invalid file path: {file_path}")
            continue
        
        output, file_type, language = process_file(file_path, lang_detector)
        print("\n--- File Analysis ---")
        print(f"File Path: {file_path}")
        print(f"File Type: {file_type}")
        print(f"Detected Language: {language}")
        print(f"Extracted Content (Snippet): {output}")
        print("---------------------\n")
