"""Naive Bayes model for sentiment classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesModel:
    """
    Multinomial Naive Bayes classifier with TF-IDF vectorization.
    
    Fast baseline model using probabilistic approach based on Bayes' theorem.
    Works well with text data but typically underperforms compared to SVM/LogReg.
    
    Attributes:
        vectorizer: TfidfVectorizer for text feature extraction
        model: MultinomialNB classifier
    """
    
    def __init__(self, ngram_range=(1, 3), max_features=10000):
        """
        Initialize Naive Bayes model with TF-IDF.
        
        Args:
            ngram_range: Range of n-grams to extract (default: (1,3) for uni/bi/trigrams)
            max_features: Maximum number of TF-IDF features (default: 10000)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'  # Remove common English stopwords
        )
        self.model = MultinomialNB(
            alpha=0.5  # Laplace smoothing parameter
        )
    
    def fit(self, X_train, y_train):
        """
        Train the Naive Bayes model.
        
        Args:
            X_train: List of training text samples
            y_train: List of training labels (1=positive, 0=negative)
        """
        # Convert texts to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        # Train Naive Bayes classifier
        self.model.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        """
        Predict sentiment labels for test data.
        
        Args:
            X_test: List of test text samples
        
        Returns:
            Array of predicted labels (1=positive, 0=negative)
        """
        # Transform test texts using fitted vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_tfidf)
