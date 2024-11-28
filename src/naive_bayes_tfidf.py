import json
import math
import re
from collections import Counter, defaultdict


def clean_text(text):
    """
    Clean and preprocess the text by converting to lowercase 
    and removing special characters.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


class NaiveBayesTFIDF:
    def __init__(self):
        self.class_word_counts = defaultdict(Counter)  # Word counts per class
        self.class_doc_counts = defaultdict(int)  # Document counts per class
        self.vocabulary = set()  # Unique vocabulary
        self.classes = set()  # Set of classes
        self.doc_freq = Counter()  # Document frequency for words
        self.total_docs = 0  # Total number of documents
        self.tfidf_cache = {}

    def compute_tfidf(self, word, category):
        """
        Compute the TF-IDF score with caching
        """
        cache_key = (word, category)
        if cache_key in self.tfidf_cache:
            return self.tfidf_cache[cache_key]

        # Original TF-IDF calculation
        word_count = self.class_word_counts[category].get(word, 0)
        total_words = sum(self.class_word_counts[category].values())
        tf = word_count / total_words if total_words > 0 else 0
        
        doc_freq = self.doc_freq[word]
        idf = math.log((self.total_docs + 1) / (1 + doc_freq)) + 1

        result = tf * idf
        self.tfidf_cache[cache_key] = result
        return result

    def train(self, training_data):
        """
        Train the Naive Bayes classifier using TF-IDF.
        """
        self.total_docs = len(training_data)

        # Count document frequencies and per-class word frequencies
        for article in training_data:
            category = article['category']
            text = clean_text(article['headline'] + ' ' + article['short_description'])
            words = text.split()

            self.classes.add(category)
            self.class_doc_counts[category] += 1

            unique_words = set(words)
            for word in unique_words:
                self.doc_freq[word] += 1
                self.vocabulary.add(word)

            # Count word occurrences per class
            self.class_word_counts[category].update(words)

    def predict(self, text):
        """
        Optimized predict function
        """
        text = clean_text(text)
        words = set(text.split())  # Using set to eliminate duplicates
        total_docs = sum(self.class_doc_counts.values())

        class_scores = {}
        for category in self.classes:
            prior = math.log(self.class_doc_counts[category] / total_docs)
            
            # Calculer la probabilité seulement pour les mots présents dans le vocabulaire
            likelihood = sum(
                math.log(self.compute_tfidf(word, category) + 1e-8)
                for word in words
                if word in self.vocabulary
            )
            
            class_scores[category] = prior + likelihood

        return max(class_scores, key=class_scores.get)


def main():
    """
    Main function to run Naive Bayes with TF-IDF.
    """
    # Load data from JSON file
    json_file = "data/articles.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Split data into training and testing
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    training_data = data[:split_index]
    testing_data = data[split_index:]

    # Initialize and train the classifier
    classifier = NaiveBayesTFIDF()
    classifier.train(training_data)

    # Test the classifier
    correct_predictions = 0
    total_predictions = len(testing_data)

    print("Naive Bayes with TF-IDF Classification Results:")
    for i, article in enumerate(testing_data):
        if i % 100 == 0:  # Afficher la progression tous les 100 articles
            print(f"Processing article {i}/{len(testing_data)}")
            
        text = article['headline'] + ' ' + article['short_description']
        predicted_category = classifier.predict(text)
        actual_category = article['category']

        if predicted_category == actual_category:
            correct_predictions += 1

        print(f"Headline: {article['headline']}")
        print(f"Predicted Category: {predicted_category}")
        print(f"Actual Category: {actual_category}")
        print("---")

    # Afficher seulement le résultat final
    accuracy = correct_predictions / total_predictions * 100
    print(f"\nAccuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
