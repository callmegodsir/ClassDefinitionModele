import json
import random
import re
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple

class NaiveBayesBinarization:
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Naive Bayes Classifier with Binarization
        
        Args:
            alpha (float): Laplace smoothing parameter (default: 1.0)
        """
        self.alpha = alpha
        self.class_probs = {}  # P(class)
        self.word_probs = {}   # P(word|class)
        self.classes = set()
        self.vocab = set()
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by cleaning and tokenizing
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            List[str]: List of unique tokens
        """ 
        # Lowercase conversion
        text = text.lower()
        # Punctuation removal
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenization and unique tokens (binarization)
        return list(set(text.split()))
    
    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the classifier on the data
        
        Args:
            texts (List[str]): Training texts
            labels (List[str]): Corresponding labels
        """
        # Count class occurrences
        class_counts = Counter(labels)
        self.classes = set(labels)
        
        # Initialize word counters by class
        word_counts = defaultdict(lambda: defaultdict(int))
        
        # Count words for each class
        for text, label in zip(texts, labels):
            words = self.preprocess_text(text)
            # Update global vocabulary
            self.vocab.update(words)
            # Count unique words for this class
            for word in words:
                word_counts[label][word] += 1
        
        # Calculate P(class)
        total_docs = len(labels)
        self.class_probs = {c: count/total_docs for c, count in class_counts.items()}
        
        # Calculate P(word|class) with Laplace smoothing
        self.word_probs = {}
        vocab_size = len(self.vocab)
        
        for c in self.classes:
            # Total unique words in the class
            total_words = len(word_counts[c])
            self.word_probs[c] = {}
            
            # Calculate probabilities for each word
            for word in self.vocab:
                count = word_counts[c][word]
                # Apply Laplace smoothing
                self.word_probs[c][word] = (count + self.alpha) / (total_words + self.alpha * vocab_size)
    
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
            words = self.preprocess_text(text)
            scores = {}
            
            # Calculate score for each class
            for c in self.classes:
                # Start with log(P(class))
                score = math.log(self.class_probs[c])
                
                # Add log(P(word|class)) for each word
                for word in words:
                    if word in self.vocab:
                        score += math.log(self.word_probs[c][word])
                
                scores[c] = score
            
            # Select class with maximum score
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
    # Calculate accuracy
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    
    # Calculate per-class metrics
    classes = set(y_true)
    metrics = {
        'accuracy': accuracy,
        'per_class': {}
    }
    
    for c in classes:
        # True positives, false positives, false negatives
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == c and pred == c)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != c and pred == c)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == c and pred != c)
        
        # Calculate precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['per_class'][c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

def main():
    """
    Main function to run the Naive Bayes classifier
    """
    # 1. Load and shuffle data
    print("Loading and shuffling data...")
    texts, labels = load_and_shuffle_data('data/articles.json')
    
    # 2. Split train/test (80/20)
    train_size = int(0.8 * len(texts))
    X_train = texts[:train_size]
    y_train = labels[:train_size]
    X_test = texts[train_size:]
    y_test = labels[train_size:]
    
    # 3. Train
    print("Training model...")
    classifier = NaiveBayesBinarization(alpha=1.0)
    classifier.fit(X_train, y_train)
    
    # 4. Predict
    print("Evaluating model...")
    predictions = classifier.predict(X_test)
    
    # 5. Evaluate
    metrics = evaluate(y_test, predictions)
    
    # 6. Display results
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print("\nPer-class Metrics:")
    for classe, scores in metrics['per_class'].items():
        print(f"\n{classe}:")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall: {scores['recall']:.4f}")
        print(f"  F1-score: {scores['f1']:.4f}")
    
    # 7. Example prediction
    print("\nExample Prediction:")
    test_text = "New documentary explores immigration and family relationships"
    prediction = classifier.predict([test_text])[0]
    print(f"Text: {test_text}")
    print(f"Predicted Category: {prediction}")

if __name__ == "__main__":
    main()