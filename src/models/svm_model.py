"""Support Vector Machine (SVM) model for sentiment classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import numpy as np


class SVMModel:
    """
    Linear Support Vector Machine classifier with TF-IDF vectorization.
    
    Uses LinearSVC which is optimized for large-scale text classification.
    Typically achieves top performance on sentiment analysis tasks.
    
    Attributes:
        vectorizer: TfidfVectorizer for text-to-features conversion
        model: LinearSVC classifier with optimized hyperparameters
    """
    
    def __init__(self, ngram_range=(1, 2), max_features=50000):
        """
        Initialize SVM model with TF-IDF vectorization.
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams + bigrams)
            max_features: Maximum number of features (default: 50000)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Keep stopwords for sentiment signals
            min_df=2,  # Ignore terms appearing in less than 2 documents
            dtype=np.float32,  # Memory optimization
            sublinear_tf=True  # Use log-scale term frequency
        )
        self.model = LinearSVC(
            C=0.3,  # Regularization strength (lower = more regularization)
            max_iter=2000  # Maximum iterations for convergence
        )
    
    def fit(self, X_train, y_train):
        """
        Train the SVM model on provided data.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (1=positive, 0=negative)
        """
        # Convert text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        # Train SVM classifier
        self.model.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        """
        Predict sentiment labels for test data.
        
        Args:
            X_test: List of test text samples
        
        Returns:
            Array of predicted labels (1=positive, 0=negative)
        """
        # Transform test data using fitted vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_tfidf)