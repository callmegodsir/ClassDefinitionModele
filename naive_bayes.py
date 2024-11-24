# naive_bayes.py

import json
import re
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        """
        Initialise le classificateur Naive Bayes
        
        Args:
            alpha (float): Paramètre de lissage de Laplace (default: 1.0)
        """
        self.alpha = alpha
        self.class_probs = {}  # P(classe)
        self.word_probs = {}   # P(mot|classe)
        self.classes = set()
        self.vocab = set()
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Prétraite le texte en le nettoyant et en le tokenisant
        
        Args:
            text (str): Texte à prétraiter
            
        Returns:
            List[str]: Liste de tokens
        """
        # Conversion en minuscules
        text = text.lower()
        # Suppression de la ponctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenization simple par espace
        return text.split()
    
    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Entraîne le classificateur sur les données
        
        Args:
            texts (List[str]): Liste des textes d'entraînement
            labels (List[str]): Liste des étiquettes correspondantes
        """
        # Comptage des classes
        class_counts = Counter(labels)
        self.classes = set(labels)
        
        # Initialisation des compteurs de mots par classe
        word_counts = defaultdict(lambda: defaultdict(int))
        
        # Comptage des mots pour chaque classe
        for text, label in zip(texts, labels):
            words = self.preprocess_text(text)
            # Mise à jour du vocabulaire global
            self.vocab.update(words)
            # Comptage des mots pour cette classe
            for word in words:
                word_counts[label][word] += 1
        
        # Calcul de P(classe)
        total_docs = len(labels)
        self.class_probs = {c: count/total_docs for c, count in class_counts.items()}
        
        # Calcul de P(mot|classe) avec lissage de Laplace
        self.word_probs = {}
        vocab_size = len(self.vocab)
        
        for c in self.classes:
            # Nombre total de mots dans la classe
            total_words = sum(word_counts[c].values())
            self.word_probs[c] = {}
            
            # Calcul des probabilités pour chaque mot
            for word in self.vocab:
                count = word_counts[c][word]
                # Application du lissage de Laplace
                self.word_probs[c][word] = (count + self.alpha) / (total_words + self.alpha * vocab_size)
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Prédit les classes pour une liste de textes
        
        Args:
            texts (List[str]): Liste des textes à classifier
            
        Returns:
            List[str]: Liste des classes prédites
        """
        predictions = []
        
        for text in texts:
            words = self.preprocess_text(text)
            scores = {}
            
            # Calcul du score pour chaque classe
            for c in self.classes:
                # Commencer avec log(P(classe))
                score = math.log(self.class_probs[c])
                
                # Ajouter log(P(mot|classe)) pour chaque mot
                for word in words:
                    if word in self.vocab:
                        score += math.log(self.word_probs[c][word])
                
                scores[c] = score
            
            # Sélectionner la classe avec le score maximum
            predictions.append(max(scores.items(), key=lambda x: x[1])[0])
        
        return predictions

def load_data(json_file: str) -> Tuple[List[str], List[str]]:
    """
    Charge les données depuis un fichier JSON
    
    Args:
        json_file (str): Chemin vers le fichier JSON
        
    Returns:
        Tuple[List[str], List[str]]: Textes et leurs étiquettes
    """
    texts = []
    labels = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Combiner le titre et la description
            text = f"{item['headline']} {item['short_description']}"
            texts.append(text)
            labels.append(item['category'])
    
    return texts, labels

def evaluate(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Évalue les performances du classificateur
    
    Args:
        y_true (List[str]): Vraies étiquettes
        y_pred (List[str]): Étiquettes prédites
        
    Returns:
        Dict[str, float]: Métriques d'évaluation
    """
    # Calcul de l'accuracy
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    
    # Calcul des métriques par classe
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
        
        # Calcul precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['per_class'][c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

if __name__ == "__main__":
    # Exemple d'utilisation
    
    # 1. Chargement des données
    print("Chargement des données...")
    texts, labels = load_data('articles.json')
    
    # 2. Séparation train/test (80/20)
    train_size = int(0.8 * len(texts))
    X_train = texts[:train_size]
    y_train = labels[:train_size]
    X_test = texts[train_size:]
    y_test = labels[train_size:]
    
    # 3. Entraînement
    print("Entraînement du modèle...")
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(X_train, y_train)
    
    # 4. Prédiction
    print("Évaluation du modèle...")
    predictions = classifier.predict(X_test)
    
    # 5. Évaluation
    metrics = evaluate(y_test, predictions)
    
    # 6. Affichage des résultats
    print(f"\nAccuracy globale: {metrics['accuracy']:.4f}")
    print("\nMétriques par classe:")
    for classe, scores in metrics['per_class'].items():
        print(f"\n{classe}:")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall: {scores['recall']:.4f}")
        print(f"  F1-score: {scores['f1']:.4f}")
    
    # 7. Exemple de prédiction sur un nouveau texte
    print("\nExemple de prédiction:")
    test_text = "New documentary explores immigration and family relationships"
    prediction = classifier.predict([test_text])[0]
    print(f"Texte: {test_text}")
    print(f"Catégorie prédite: {prediction}")