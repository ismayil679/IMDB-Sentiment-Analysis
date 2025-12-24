"""Generate predictions for confusion matrix visualization.

Trains top 2 models (SVM and Logistic Regression) and saves their predictions
to data/predictions/ directory for later confusion matrix generation.

Run this once after experiments to enable confusion matrix visualization:
    python generate_predictions.py
    
Then generate visualizations:
    python src/visualize.py
"""

import sys
import os

# Add src/ to Python path for imports
sys.path.insert(0, 'src')

from load_data import load_dataset
from preprocess import preprocess_texts
from models.svm_model import SVMModel
from models.logistic_regression_model import LogisticRegressionModel
from utils import save_predictions


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    
    # Load dataset with stratified 60/20/20 split
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("data/IMDB Dataset.csv")
    
    # Apply preprocessing (lowercase + HTML removal)
    X_train = preprocess_texts(X_train)
    X_val = preprocess_texts(X_val)
    X_test = preprocess_texts(X_test)
    
    # Combine train and validation for final model training
    X_train_full = X_train + X_val
    y_train_full = y_train + y_val
    
    # Ensure predictions directory exists
    os.makedirs("data/predictions", exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS FOR TOP 2 MODELS")
    print("="*70)
    
    # ========================================================================
    # MODEL 1: SVM (Best F1 Score)
    # ========================================================================
    print("\n[1/2] Training SVM (1,2) n-grams...")
    svm_model = SVMModel(ngram_range=(1, 2))
    svm_model.fit(X_train_full, y_train_full)
    y_pred_svm = svm_model.predict(X_test)
    save_predictions("SVM", (1, 2), y_test, y_pred_svm)
    print("      ✓ SVM predictions saved to data/predictions/")
    
    # ========================================================================
    # MODEL 2: Logistic Regression (2nd Best F1 Score)
    # ========================================================================
    print("\n[2/2] Training Logistic Regression (1,2) n-grams...")
    lr_model = LogisticRegressionModel(ngram_range=(1, 2))
    lr_model.fit(X_train_full, y_train_full)
    y_pred_lr = lr_model.predict(X_test)
    save_predictions("Logistic Regression", (1, 2), y_test, y_pred_lr)
    print("      ✓ Logistic Regression predictions saved to data/predictions/")
    
    print("\n" + "="*70)
    print("✓ PREDICTIONS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: Generate visualizations")
    print("  python src/visualize.py")
