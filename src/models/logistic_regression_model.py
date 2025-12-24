"""Logistic Regression model for sentiment classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import numpy as np


class LogisticRegressionModel(BaseEstimator):
    """
    Logistic Regression classifier with elastic net regularization and TF-IDF.
    
    Uses sklearn Pipeline for integrated vectorization and classification.
    Elastic net combines L1 (sparsity) and L2 (stability) regularization.
    
    Attributes:
        ngram_range: N-gram extraction range
        max_features: Maximum TF-IDF features
        model: Pipeline containing vectorizer and classifier
    """
    
    def __init__(self, ngram_range=(1, 2), max_features=50000):
        """
        Initialize Logistic Regression model with TF-IDF pipeline.
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams + bigrams)
            max_features: Maximum number of TF-IDF features (default: 50000)
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        
        # Create pipeline: TF-IDF vectorization -> Logistic Regression
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=None,  # Preserve sentiment-bearing stopwords
                min_df=2,  # Ignore rare terms
                dtype=np.float32,
                sublinear_tf=True  # Apply log scaling to term frequency
            )),
            ('classifier', LogisticRegression(
                C=4.0,  # Inverse regularization strength (higher = less regularization)
                penalty='elasticnet',  # L1 + L2 regularization
                l1_ratio=0.2,  # 20% L1, 80% L2
                solver='saga',  # Solver supporting elastic net
                max_iter=2000,  # Maximum iterations
                n_jobs=-1  # Use all CPU cores
            ))
        ])
    
    def fit(self, X_train, y_train):
        """
        Train the Logistic Regression model.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (1=positive, 0=negative)
        
        Returns:
            self: Trained model instance
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """
        Predict sentiment labels for test data.
        
        Args:
            X_test: List of test text samples
        
        Returns:
            Array of predicted labels (1=positive, 0=negative)
        """
        return self.model.predict(X_test)
