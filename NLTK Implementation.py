!pip install nltk==3.8.1
import re
import nltk
from collections import defaultdict

# Download stopwords and 'punkt'
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

class Rake:
    def __init__(self, stop_words):
        self.stop_words = set(stop_words)

    def split_sentences(self, text):
        sentence_delimiters = re.compile(r'[.!?,;:\t"\'()\u2019\u2013-]')
        sentences = sentence_delimiters.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence.lower())
            phrase = []
            for word in words:
                if word in self.stop_words or not word.isalnum():
                    if phrase:
                        phrase_list.append(" ".join(phrase))
                        phrase = []
                else:
                    phrase.append(word)
            if phrase:
                phrase_list.append(" ".join(phrase))
        return phrase_list

    def calculate_word_scores(self, phrase_list):
        word_freq = defaultdict(int)
        word_degree = defaultdict(int)
        for phrase in phrase_list:
            words = phrase.split()
            word_list_length = len(words)
            word_list_degree = word_list_length - 1
            for word in words:
                word_freq[word] += 1
                word_degree[word] += word_list_degree

        # Compute word scores
        word_scores = {}
        for word in word_freq:
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-8
            word_scores[word] = (word_degree[word] + word_freq[word]) / (word_freq[word] + epsilon)
        return word_scores

    def generate_keyword_scores(self, phrase_list, word_scores):
        keyword_candidates = defaultdict(float)
        for phrase in phrase_list:
            words = phrase.split()
            candidate_score = 0
            for word in words:
                candidate_score += word_scores.get(word, 0)
            if candidate_score > 0:  # Only consider phrases with a positive score
                keyword_candidates[phrase] = candidate_score
        return keyword_candidates

    def extract_keywords(self, text):
        sentences = self.split_sentences(text)
        phrase_list = self.generate_candidate_keywords(sentences)
        word_scores = self.calculate_word_scores(phrase_list)
        keyword_candidates = self.generate_keyword_scores(phrase_list, word_scores)
        sorted_keywords = sorted(keyword_candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords

# Example usage
if __name__ == "__main__":
    text = """
    The Eiffel Tower is one of the most famous landmarks in the world.
    Located in Paris, France, it attracts millions of tourists every year.
    Built in 1889, it was initially criticized by some of France's leading artists and intellectuals for its design,
    but it has become a global cultural icon of France and one of the most recognizable structures in the world.
    The tower is 330 meters tall and offers visitors stunning views of Paris.
    """

    stop_words = stopwords.words('english')
    additional_stopwords = ['could', 'be', 'the']
    stop_words.extend(additional_stopwords)

    rake = Rake(stop_words)

    # Extract keywords
    keywords = rake.extract_keywords(text)

    # Print all keywords and their scores
    print("Extracted Keywords and their scores:")
    for keyword, score in keywords:
        print(f"{keyword}: {score:.2f}")

    threshold = 4.0
    filtered_keywords = [keyword for keyword, score in keywords if score >= threshold] # Use keywords instead of ranked_phrases

    print("\nFiltered Keywords (Score Threshold = 4.0):")
    print(filtered_keywords)
