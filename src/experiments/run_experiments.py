from load_data import load_dataset
from preprocess import preprocess_texts
from models.logistic_regression_model import LogisticRegressionModel
from models.naive_bayes_model import NaiveBayesModel
from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from utils import evaluate_model, save_results
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_all_experiments(csv_path):
    # Clear previous results to avoid duplicates
    results_file = 'data/results.csv'
    if os.path.exists(results_file):
        os.remove(results_file)
    
    # Load and preprocess
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(csv_path)
    X_train = preprocess_texts(X_train)
    X_val = preprocess_texts(X_val)
    X_test = preprocess_texts(X_test)
    
    # Combine train and val for final training
    X_train_full = X_train + X_val
    y_train_full = y_train + y_val

    experiments = [
        # Logistic Regression
        {"model_class": LogisticRegressionModel, "ngram_range": (1,2), "name": "Logistic Regression"},
        # Naive Bayes
        # {"model_class": NaiveBayesModel, "ngram_range": (1,2), "name": "Naive Bayes"},
        # {"model_class": NaiveBayesModel, "ngram_range": (1,3), "name": "Naive Bayes"},
        # SVM
        {"model_class": SVMModel, "ngram_range": (1,2), "name": "SVM"},
        # Tree-based models
        # {"model_class": CatBoostModel, "ngram_range": (1,2), "name": "CatBoost"},
        # {"model_class": LightGBMModel, "ngram_range": (1,2), "name": "LightGBM"},
        # {"model_class": RandomForestModel, "ngram_range": (1,2), "name": "Random Forest"},
        # {"model_class": XGBoostModel, "ngram_range": (1,2), "name": "XGBoost"},
        # You can add more algorithms here later
    ]

    for exp in experiments:
        print(f"\nRunning {exp['name']} with {exp['ngram_range']} n-grams...")
        model = exp["model_class"](ngram_range=exp["ngram_range"])
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        print(f"{exp['name']} results: Accuracy={metrics[0]:.5f}, F1={metrics[3]:.5f}")
        save_results(exp["name"], exp["ngram_range"], metrics)
