"""CatBoost model for sentiment classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator


class CatBoostModel(BaseEstimator):
    """
    CatBoost gradient boosting classifier with TF-IDF vectorization.
    
    Yandex's gradient boosting library with ordered boosting algorithm.
    Often handles high-dimensional sparse features well but slower for text.
    
    Attributes:
        vectorizer: TfidfVectorizer for text feature extraction
        classifier: CatBoostClassifier with default parameters
    """
    
    def __init__(self, ngram_range=(1, 2), max_features=10000):
        """
        Initialize CatBoost model with TF-IDF.
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams + bigrams)
            max_features: Maximum number of TF-IDF features (default: 10000)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'  # Remove stopwords
        )
        self.classifier = CatBoostClassifier(
            iterations=100,  # Number of boosting iterations
            depth=6,  # Tree depth
            random_state=42,  # For reproducibility
            verbose=False  # Suppress training output
        )
    
    def fit(self, X_train, y_train):
        """
        Train the CatBoost model.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (1=positive, 0=negative)
        
        Returns:
            self: Trained model instance
        """
        # Convert texts to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        # Train CatBoost classifier
        self.classifier.fit(X_train_tfidf, y_train)
        return self
    
    def predict(self, X_test):
        """
        Predict sentiment labels using CatBoost.
        
        Args:
            X_test: List of test text samples
        
        Returns:
            Array of predicted labels (1=positive, 0=negative)
        """
        # Transform test texts
        X_test_tfidf = self.vectorizer.transform(X_test)
        # Predict using trained model
        return self.classifier.predict(X_test_tfidf)