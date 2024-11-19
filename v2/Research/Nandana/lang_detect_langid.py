import langid
import logging
from datetime import datetime

class DocumentProcessorLangid:
    def __init__(self):
        self.logger = self.setup_logging()

    def setup_logging(self):
        log_file = f"processing_{datetime.now().strftime('%Y-%m-%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        return logging.getLogger(__name__)

    def analyze_languages(self, text):
        lang, confidence = langid.classify(text)
        return {lang: confidence * 100}

    def process_text(self, text):
        lang_results = self.analyze_languages(text)
        return lang_results


def main_langid():
    processor = DocumentProcessorLangid()

    while True:
        text_input = input("Enter the text for analysis (or type 'exit' to quit): ").strip()
        if text_input.lower() == 'exit':
            print("Exiting program.")
            break

        lang_results = processor.process_text(text_input)
        print("\n--- Language Detection Results ---")
        for lang, percentage in lang_results.items():
            print(f"{lang}: {round(percentage, 2)}%")
        print("----------------------------\n")

if __name__ == "__main__":
    main_langid()
