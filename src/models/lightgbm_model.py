"""LightGBM model for sentiment classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier


class LightGBMModel:
    """
    LightGBM gradient boosting classifier with TF-IDF vectorization.
    
    Microsoft's efficient gradient boosting framework optimized for speed.
    Uses leaf-wise tree growth strategy for faster training on large datasets.
    
    Attributes:
        vectorizer: TfidfVectorizer for text feature extraction
        model: LGBMClassifier with default hyperparameters
    """
    
    def __init__(self, ngram_range=(1, 2), max_features=20000):
        """
        Initialize LightGBM model with TF-IDF.
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams + bigrams)
            max_features: Maximum number of TF-IDF features (default: 20000)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'  # Remove stopwords
        )
        self.model = LGBMClassifier(
            n_estimators=200,  # Number of boosting rounds
            max_depth=6,  # Maximum tree depth
            random_state=42  # For reproducibility
        )
    
    def fit(self, X_train, y_train):
        """
        Train the LightGBM model using leaf-wise tree growth.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (1=positive, 0=negative)
        """
        # Convert texts to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        # Train LightGBM classifier
        self.model.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        """
        Predict sentiment labels using LightGBM ensemble.
        
        Args:
            X_test: List of test text samples
        
        Returns:
            Array of predicted labels (1=positive, 0=negative)
        """
        # Transform test texts
        X_test_tfidf = self.vectorizer.transform(X_test)
        # Predict using trained model
        return self.model.predict(X_test_tfidf)