import json
import random
import re
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple

class NaiveBayesNGram:
    def __init__(self, n: int = 2, alpha: float = 1.0):
        """
        Initialize the Naive Bayes Classifier with N-Grams
        
        Args:
            n (int): Size of the n-grams (default: 2)
            alpha (float): Laplace smoothing parameter (default: 1.0)
        """
        self.n = n
        self.alpha = alpha
        self.class_probs = {}  # P(class)
        self.ngram_probs = {}  # P(ngram|class)
        self.classes = set()
        self.vocab = set()
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by cleaning and generating n-grams
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            List[str]: List of n-grams
        """
        # Lowercase conversion and punctuation removal
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Tokenize
        tokens = text.split()
        # Create n-grams
        return [" ".join(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)]
    
    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the classifier on the data
        
        Args:
            texts (List[str]): Training texts
            labels (List[str]): Corresponding labels
        """
        class_counts = Counter(labels)
        self.classes = set(labels)
        
        # Initialize n-gram counters by class
        ngram_counts = defaultdict(lambda: defaultdict(int))
        
        # Count n-grams for each class
        for text, label in zip(texts, labels):
            ngrams = self.preprocess_text(text)
            self.vocab.update(ngrams)
            for ngram in ngrams:
                ngram_counts[label][ngram] += 1
        
        # Calculate P(class)
        total_docs = len(labels)
        self.class_probs = {c: count / total_docs for c, count in class_counts.items()}
        
        # Calculate P(ngram|class) with Laplace smoothing
        self.ngram_probs = {}
        vocab_size = len(self.vocab)
        for c in self.classes:
            total_ngrams = sum(ngram_counts[c].values())
            self.ngram_probs[c] = {}
            for ngram in self.vocab:
                count = ngram_counts[c][ngram]
                self.ngram_probs[c][ngram] = (count + self.alpha) / (total_ngrams + self.alpha * vocab_size)
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict classes for a list of texts
        
        Args:
            texts (List[str]): Texts to classify
            
        Returns:
            List[str]: Predicted classes
        """
        predictions = []
        for text in texts:
            ngrams = self.preprocess_text(text)
            scores = {}
            for c in self.classes:
                # Start with log(P(class))
                score = math.log(self.class_probs[c])
                # Add log(P(ngram|class)) for each n-gram
                for ngram in ngrams:
                    if ngram in self.vocab:
                        score += math.log(self.ngram_probs[c].get(ngram, self.alpha / len(self.vocab)))
                scores[c] = score
            predictions.append(max(scores.items(), key=lambda x: x[1])[0])
        return predictions

def load_and_shuffle_data(json_file: str, random_seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Load data from a JSON file and shuffle it randomly.
    
    Args:
        json_file (str): Path to the JSON file
        random_seed (int): Seed for reproducibility
        
    Returns:
        Tuple[List[str], List[str]]: Shuffled texts and their labels
    """
    texts = []
    labels = []
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Shuffle the entire dataset
    random.shuffle(data)
    
    for item in data:
        # Combine headline and description
        text = f"{item['headline']} {item['short_description']}"
        texts.append(text)
        labels.append(item['category'])
    
    return texts, labels

def evaluate(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Evaluate classifier performance
    
    Args:
        y_true (List[str]): True labels
        y_pred (List[str]): Predicted labels
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    classes = set(y_true)
    metrics = {"accuracy": accuracy, "per_class": {}}
    for c in classes:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == c and pred == c)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != c and pred == c)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == c and pred != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics["per_class"][c] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics

def main():
    """
    Main function to run Naive Bayes with N-Grams
    """
    print("Loading and shuffling data...")
    texts, labels = load_and_shuffle_data("data/articles.json")

    train_size = int(0.8 * len(texts))
    X_train, y_train = texts[:train_size], labels[:train_size]
    X_test, y_test = texts[train_size:], labels[train_size:]

    print("Training Naive Bayes with N-Grams...")
    classifier = NaiveBayesNGram(n=2, alpha=1.0)
    classifier.fit(X_train, y_train)

    print("Evaluating Naive Bayes with N-Grams...")