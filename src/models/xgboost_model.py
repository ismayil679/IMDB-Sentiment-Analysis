"""XGBoost model for sentiment classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


class XGBoostModel:
    """
    XGBoost gradient boosting classifier with TF-IDF vectorization.
    
    Advanced gradient boosting algorithm that builds trees sequentially.
    Often performs well but typically slower than linear models for high-dimensional text.
    
    Attributes:
        vectorizer: TfidfVectorizer for text feature extraction
        model: XGBClassifier with optimized hyperparameters
    """
    
    def __init__(self, ngram_range=(1, 2), max_features=20000):
        """
        Initialize XGBoost model with TF-IDF.
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams + bigrams)
            max_features: Maximum number of TF-IDF features (default: 20000)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'  # Remove stopwords
        )
        self.model = XGBClassifier(
            n_estimators=200,  # Number of boosting rounds
            max_depth=6,  # Maximum tree depth
            random_state=42,  # For reproducibility
            use_label_encoder=False,  # Deprecated parameter
            eval_metric='logloss'  # Loss function for binary classification
        )
    
    def fit(self, X_train, y_train):
        """
        Train the XGBoost model using gradient boosting.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (1=positive, 0=negative)
        """
        # Convert texts to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        # Train XGBoost with sequential tree building
        self.model.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        """
        Predict sentiment labels using gradient boosted ensemble.
        
        Args:
            X_test: List of test text samples
        
        Returns:
            Array of predicted labels (1=positive, 0=negative)
        """
        # Transform test texts
        X_test_tfidf = self.vectorizer.transform(X_test)
        # Predict using trained boosted model
        return self.model.predict(X_test_tfidf)