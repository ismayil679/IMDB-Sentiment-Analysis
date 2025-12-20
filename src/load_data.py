import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    X = df["review"].tolist()
    y = [1 if s.lower() == "positive" else 0 for s in df["sentiment"]]
    
    print(f"Total samples: {len(y)}")
    print(f"Positive samples: {sum(y)}")
    print(f"Negative samples: {len(y) - sum(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test
