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
    
    def __init__(self, ngram_range=(1, 2), max_features=30000):
        """
        Initialize Logistic Regression model with TF-IDF pipeline.
        
        OPTIMIZED VERSION (v2 - Further tuned):
        - Reduced features: 50k → 30k (faster, less memory)
        - Liblinear solver: Much faster for high-dimensional sparse data
        - L2 penalty only: Simpler and faster than elastic net
        - Early stopping: Converges faster with tolerance parameter
        - C=4.0: Validation-based tuning shows optimal regularization
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams + bigrams)
            max_features: Maximum number of TF-IDF features (default: 30000, optimized)
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        
        # Create pipeline: TF-IDF vectorization -> Logistic Regression
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=None,  # Preserve sentiment-bearing stopwords
                min_df=3,  # Ignore very rare terms (increased from 2)
                max_df=0.95,  # Ignore very common terms
                dtype=np.float32,
                sublinear_tf=True  # Apply log scaling to term frequency
            )),
            ('classifier', LogisticRegression(
                C=4.0,  # Inverse regularization strength (validation-optimized from 2.0)
                penalty='l2',  # L2 regularization (faster than elastic net)
                solver='liblinear',  # Best for high-dimensional sparse data
                max_iter=500,  # Reduced iterations (converges faster with liblinear)
                tol=1e-4,  # Early stopping tolerance
                dual=False,  # Primal formulation (better for n_samples > n_features)
                random_state=42
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
