from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, ngram_range=(1,2), max_features=20000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
        self.model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

    def fit(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)

    def predict(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_tfidf)