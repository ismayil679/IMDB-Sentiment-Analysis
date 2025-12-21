import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_data import load_dataset
from preprocess import preprocess_texts
from models.svm_model import SVMModel
from models.logistic_regression_model import LogisticRegressionModel
from models.naive_bayes_model import NaiveBayesModel
from utils import evaluate_model, save_results
from threadpoolctl import threadpool_limits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import os
import warnings
import time

# Optional tree-based models
try:
    from models.random_forest_model import RandomForestModel
    RANDOM_FOREST_AVAILABLE = True
except ImportError:
    RANDOM_FOREST_AVAILABLE = False

try:
    from models.xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from models.catboost_model import CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from models.lightgbm_model import LightGBMModel
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_all_experiments(csv_path):
    # Use 8 cores for BLAS/OpenMP operations
    with threadpool_limits(limits=8):
        # Clear previous results to avoid duplicates
        results_file = 'data/results.csv'
        if os.path.exists(results_file):
            os.remove(results_file)
        
        # Load and preprocess with stratified splits
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(csv_path)
        X_train = preprocess_texts(X_train)
        X_val = preprocess_texts(X_val)
        X_test = preprocess_texts(X_test)
        
        # Combine train and val for final training
        X_train_full = X_train + X_val
        y_train_full = y_train + y_val
        
        print("\n" + "="*70)
        print("XGBOOST HYPERPARAMETER TUNING ON VALIDATION SET")
        print("="*70)
        
        # XGBoost tuning grid - focus on key parameters
        if XGBOOST_AVAILABLE:
            xgb_candidates = [
                {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
                {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
                {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
                {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.1, "subsample": 0.8},
                {"n_estimators": 150, "max_depth": 10, "learning_rate": 0.1, "subsample": 0.8},
                {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8},
                {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.2, "subsample": 0.8},
                {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9},
            ]
            
            best_xgb_config = None
            best_xgb_val_f1 = -1
            
            from sklearn.metrics import f1_score
            
            print(f"\nTotal XGBoost candidates: {len(xgb_candidates)}")
            for idx, config in enumerate(xgb_candidates, 1):
                print(f"\n[{idx}/{len(xgb_candidates)}] n_est={config['n_estimators']}, depth={config['max_depth']}, lr={config['learning_rate']}, subsample={config['subsample']}...")
                start_time = time.time()
                
                # Train on train set only
                model = XGBoostModel(ngram_range=(1,2))
                model.model.n_estimators = config['n_estimators']
                model.model.max_depth = config['max_depth']
                model.model.learning_rate = config['learning_rate']
                model.model.subsample = config['subsample']
                model.fit(X_train, y_train)
                
                # Validate
                y_val_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred)
                elapsed = time.time() - start_time
                
                print(f"  Val F1={val_f1:.5f}, Time={elapsed:.2f}s")
                
                if val_f1 > best_xgb_val_f1:
                    best_xgb_val_f1 = val_f1
                    best_xgb_config = config
                    print(f"  ✓ New best!")
            
            print(f"\n{'='*70}")
            print(f"BEST XGBOOST: n_est={best_xgb_config['n_estimators']}, depth={best_xgb_config['max_depth']}, lr={best_xgb_config['learning_rate']}, subsample={best_xgb_config['subsample']}")
            print(f"Validation F1: {best_xgb_val_f1:.5f}")
            print(f"{'='*70}\n")
            
            # Test best config on test set
            print("\n=== Testing Best XGBoost on Test Set ===")
            for ngram in [(1,2), (1,3)]:
                print(f"\nTraining XGBoost with {ngram} n-grams...")
                start_time = time.time()
                
                model = XGBoostModel(ngram_range=ngram)
                model.model.n_estimators = best_xgb_config['n_estimators']
                model.model.max_depth = best_xgb_config['max_depth']
                model.model.learning_rate = best_xgb_config['learning_rate']
                model.model.subsample = best_xgb_config['subsample']
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
                elapsed = time.time() - start_time
                metrics = evaluate_model(y_test, y_pred)
                print(f"XGBoost - Accuracy: {metrics[0]:.5f}, F1: {metrics[3]:.5f}, Time: {elapsed:.2f}s")
                save_results("XGBoost", ngram, metrics)
        else:
            print("\nXGBoost not available. Install with: pip install xgboost")
        
        # Commented out other models for speed during tree model tuning
        """
        # Define experiments with different n-gram configurations
        experiments = [
            # Logistic Regression variants
            {"name": "LogReg (Elasticnet)", "type": "logreg_elasticnet", "ngram": (1,2)},
            {"name": "LogReg (Elasticnet)", "type": "logreg_elasticnet", "ngram": (1,3)},
            {"name": "LogReg (L2)", "type": "logreg_l2", "ngram": (1,2)},
            {"name": "LogReg (L2)", "type": "logreg_l2", "ngram": (1,3)},
            # SVM
            {"name": "SVM", "type": "svm", "ngram": (1,2)},
            {"name": "SVM", "type": "svm", "ngram": (1,3)},
            # Naive Bayes
            {"name": "Naive Bayes", "type": "naive_bayes", "ngram": (1,2)},
            {"name": "Naive Bayes", "type": "naive_bayes", "ngram": (1,3)},
        ]
        
        # Add optional tree-based models
        if RANDOM_FOREST_AVAILABLE:
            experiments.extend([
                {"name": "Random Forest", "type": "random_forest", "ngram": (1,2)},
                {"name": "Random Forest", "type": "random_forest", "ngram": (1,3)},
            ])
        if XGBOOST_AVAILABLE:
            experiments.extend([
                {"name": "XGBoost", "type": "xgboost", "ngram": (1,2)},
                {"name": "XGBoost", "type": "xgboost", "ngram": (1,3)},
            ])
        if CATBOOST_AVAILABLE:
            experiments.extend([
                {"name": "CatBoost", "type": "catboost", "ngram": (1,2)},
                {"name": "CatBoost", "type": "catboost", "ngram": (1,3)},
            ])
        if LIGHTGBM_AVAILABLE:
            experiments.extend([
                {"name": "LightGBM", "type": "lightgbm", "ngram": (1,2)},
                {"name": "LightGBM", "type": "lightgbm", "ngram": (1,3)},
            ])
        
        # Run all experiments
        for exp in experiments:
            print(f"\nTraining {exp['name']} with {exp['ngram']} n-grams...")
            start_time = time.time()
            
            if exp['type'] == 'logreg_elasticnet':
                vectorizer = TfidfVectorizer(
                    max_features=60000,
                    ngram_range=exp['ngram'],
                    stop_words=None,
                    min_df=2,
                    dtype=np.float32,
                    sublinear_tf=True
                )
                classifier = LogisticRegression(
                    C=4.0,
                    penalty='elasticnet',
                    l1_ratio=0.1,
                    solver='saga',
                    max_iter=1000,
                    n_jobs=-1
                )
                model = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
            elif exp['type'] == 'logreg_l2':
                vectorizer = TfidfVectorizer(
                    max_features=60000,
                    ngram_range=exp['ngram'],
                    stop_words=None,
                    min_df=2,
                    dtype=np.float32,
                    sublinear_tf=True
                )
                classifier = LogisticRegression(
                    C=4.0,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000
                )
                model = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
            elif exp['type'] == 'svm':
                model = SVMModel(ngram_range=exp['ngram'], max_features=70000)
                model.model.C = 0.4
                model.model.max_iter = 1500
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
            elif exp['type'] == 'naive_bayes':
                model = NaiveBayesModel(ngram_range=exp['ngram'])
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
            elif exp['type'] == 'random_forest':
                model = RandomForestModel(ngram_range=exp['ngram'])
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
            elif exp['type'] == 'xgboost':
                model = XGBoostModel(ngram_range=exp['ngram'])
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
            elif exp['type'] == 'catboost':
                model = CatBoostModel(ngram_range=exp['ngram'])
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
                
            elif exp['type'] == 'lightgbm':
                model = LightGBMModel(ngram_range=exp['ngram'])
                model.fit(X_train_full, y_train_full)
                y_pred = model.predict(X_test)
            
            elapsed = time.time() - start_time
            metrics = evaluate_model(y_test, y_pred)
            print(f"{exp['name']} - Accuracy: {metrics[0]:.5f}, F1: {metrics[3]:.5f}, Time: {elapsed:.2f}s")
            save_results(exp['name'], exp['ngram'], metrics)
        """

        print("\n" + "="*70)
        print("All experiments completed! Results saved to data/results.csv")
        print("="*70)
        
        # Read and display results
        import pandas as pd
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            print("\n=== Final Results ===")
            print(results_df.to_string(index=False))
