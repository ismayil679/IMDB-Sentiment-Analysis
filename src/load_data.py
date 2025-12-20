import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    X = df["review"].tolist()
    y = [1 if s.lower() == "positive" else 0 for s in df["sentiment"]]
    
    print(f"Total samples: {len(y)}")
    print(f"Positive samples: {sum(y)}")
    print(f"Negative samples: {len(y) - sum(y)}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
