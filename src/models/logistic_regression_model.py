from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

class LogisticRegressionModel(BaseEstimator):
    def __init__(self, ngram_range=(1,2), max_features=50000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=None)),
            ('classifier', LogisticRegression(C=4.0, penalty='elasticnet', l1_ratio=0.2, solver='saga', max_iter=2000))
        ])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)
