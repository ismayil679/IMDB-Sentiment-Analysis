from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator

class CatBoostModel(BaseEstimator):
    def __init__(self, ngram_range=(1,2), max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
        self.classifier = CatBoostClassifier(iterations=100, depth=6, random_state=42, verbose=False)

    def fit(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)
        return self

    def predict(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_tfidf)