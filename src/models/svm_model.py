from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

class SVMModel:
    def __init__(self, ngram_range=(1,2), max_features=50000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=None)
        self.model = LinearSVC(C=0.3, max_iter=2000)

    def fit(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)

    def predict(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_tfidf)