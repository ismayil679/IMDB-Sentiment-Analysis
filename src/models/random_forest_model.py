"""Random Forest model for sentiment classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    """
    Random Forest ensemble classifier with TF-IDF vectorization.
    
    Tree-based ensemble model that builds multiple decision trees.
    Typically slower and less accurate than linear models (SVM/LogReg) for text,
    but useful for comparison and understanding ensemble methods.
    
    Attributes:
        vectorizer: TfidfVectorizer for text feature extraction
        model: RandomForestClassifier with 200 trees
    """
    
    def __init__(self, ngram_range=(1, 2), max_features=20000):
        """
        Initialize Random Forest model with TF-IDF.
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,2) for unigrams + bigrams)
            max_features: Maximum number of TF-IDF features (default: 20000)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'  # Remove stopwords for cleaner tree splits
        )
        self.model = RandomForestClassifier(
            n_estimators=200,  # Number of trees in the forest
            max_depth=20,  # Maximum tree depth
            random_state=42  # For reproducibility
        )
    
    def fit(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (1=positive, 0=negative)
        """
        # Convert texts to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        # Train Random Forest (builds 200 decision trees)
        self.model.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        """
        Predict sentiment labels using ensemble voting.
        
        Args:
            X_test: List of test text samples
        
        Returns:
            Array of predicted labels (1=positive, 0=negative)
        """
        # Transform test texts
        X_test_tfidf = self.vectorizer.transform(X_test)
        # Predict by majority vote across all trees
        return self.model.predict(X_test_tfidf)