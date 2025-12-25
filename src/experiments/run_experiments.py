"""
IMDB Sentiment Analysis - Model Experiments Module

This module runs hyperparameter tuning and evaluation for multiple ML models:
- Linear Models: SVM, Logistic Regression (best performers)
- Baseline: Naive Bayes
- Tree Models: Random Forest, XGBoost, CatBoost, LightGBM

Workflow:
1. Load and preprocess data with stratified 60/20/20 split
2. Train all available models on train+validation sets
3. Evaluate on held-out test set
4. Save results with timing and predictions for analysis

Results are saved to: data/results.csv
Predictions are saved to: data/predictions/
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Core imports
from load_data import load_dataset
from preprocess import preprocess_texts
from utils import evaluate_model, save_results, save_predictions
from threadpoolctl import threadpool_limits
from sklearn.metrics import f1_score

# Model imports (required)
from models.svm_model import SVMModel
from models.logistic_regression_model import LogisticRegressionModel
from models.naive_bayes_model import NaiveBayesModel
from models.ensemble_model import EnsembleModel

# Optional model imports with graceful fallbacks
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

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')



def run_all_experiments(csv_path):
    """
    Run all model experiments with proper train/val/test evaluation.
    
    Workflow:
    1. Load and preprocess dataset with stratified splits (60/20/20)
    2. Run each model with specified n-gram configurations
    3. Track training time for performance analysis
    4. Track experiments with MLflow for reproducibility
    5. Save predictions for confusion matrix generation
    6. Display final results ranking
    
    Args:
        csv_path: Path to IMDB Dataset CSV file
    """
    
    # Configure MLflow
    mlflow.set_experiment("IMDB_Sentiment_Analysis")
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Use 8 cores for BLAS/OpenMP operations
    with threadpool_limits(limits=8):
        
        print("\n" + "="*80)
        print("IMDB SENTIMENT ANALYSIS - EXPERIMENTAL EVALUATION")
        print("="*80)
        print(f"MLflow Tracking: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {mlflow.get_experiment_by_name('IMDB_Sentiment_Analysis').name}")
        
        # Clear previous results to avoid duplicates
        results_file = 'data/results.csv'
        if os.path.exists(results_file):
            os.remove(results_file)
            print("✓ Cleared previous results")
        
        # Load dataset with stratified 60/20/20 split
        print("\nLoading dataset...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(csv_path)
        
        # Preprocess texts (lowercase + HTML removal)
        print("Preprocessing texts...")
        X_train = preprocess_texts(X_train)
        X_val = preprocess_texts(X_val)
        X_test = preprocess_texts(X_test)
        
        # Combine train+val for final model training (best practice)
        X_train_full = X_train + X_val
        y_train_full = y_train + y_val
        
        print(f"✓ Data ready: Train={len(X_train_full)}, Test={len(X_test)}\n")
        
        # ============================================================================
        # DEFINE EXPERIMENTS
        # ============================================================================
        
        # All experiments with model type and n-gram configurations
        experiments = [
            # ---- LINEAR MODELS (Best Performers) ----
            {"name": "SVM", "type": "svm", "ngram": (1, 2)},
            {"name": "SVM", "type": "svm", "ngram": (1, 3)},
            
            {"name": "Logistic Regression", "type": "logreg", "ngram": (1, 2)},
            {"name": "Logistic Regression", "type": "logreg", "ngram": (1, 3)},
            
            # ---- BASELINE MODEL ----
            {"name": "Naive Bayes", "type": "naive_bayes", "ngram": (1, 2)},
            {"name": "Naive Bayes", "type": "naive_bayes", "ngram": (1, 3)},
        ]
        
        # Add optional tree-based models if available
        if RANDOM_FOREST_AVAILABLE:
            experiments.extend([
                {"name": "Random Forest", "type": "random_forest", "ngram": (1, 2)},
                {"name": "Random Forest", "type": "random_forest", "ngram": (1, 3)},
            ])
        
        if XGBOOST_AVAILABLE:
            experiments.extend([
                {"name": "XGBoost", "type": "xgboost", "ngram": (1, 2)},
                {"name": "XGBoost", "type": "xgboost", "ngram": (1, 3)},
            ])
        
        if CATBOOST_AVAILABLE:
            experiments.extend([
                {"name": "CatBoost", "type": "catboost", "ngram": (1, 2)},
                {"name": "CatBoost", "type": "catboost", "ngram": (1, 3)},
            ])
        
        if LIGHTGBM_AVAILABLE:
            experiments.extend([
                {"name": "LightGBM", "type": "lightgbm", "ngram": (1, 2)},
                {"name": "LightGBM", "type": "lightgbm", "ngram": (1, 3)},
            ])
        
        print(f"Total experiments to run: {len(experiments)}")
        print("-" * 80)
        
        # ============================================================================
        # RUN EXPERIMENTS
        # ============================================================================
        
        for idx, exp in enumerate(experiments, 1):
            model_name = exp['name']
            ngram_range = exp['ngram']
            model_type = exp['type']
            ngram_str = f"{ngram_range[0]}-{ngram_range[1]}"
            
            print(f"[{idx:2d}/{len(experiments)}] {model_name:20s} ({ngram_str})... ", end="", flush=True)
            
            # Start MLflow run for this experiment
            with mlflow.start_run(run_name=f"{model_name}_{ngram_str}"):
                try:
                    # Log parameters
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("ngram_range", ngram_str)
                    mlflow.log_param("train_samples", len(X_train_full))
                    mlflow.log_param("test_samples", len(X_test))
                    
                    start_time = time.time()
                    
                    # Train and evaluate based on model type
                    if model_type == 'ensemble':
                        model = EnsembleModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "ensemble")
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                    
                    elif model_type == 'svm':
                        model = SVMModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "SVM")
                        mlflow.log_param("C", 0.35)
                        mlflow.log_param("max_features", 90000)
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                    
                    elif model_type == 'logreg':
                        model = LogisticRegressionModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "Logistic Regression")
                        mlflow.log_param("C", 4.0)
                        mlflow.log_param("solver", "liblinear")
                        mlflow.log_param("max_features", 30000)
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                
                elif model_type == 'naive_bayes':
                    model = NaiveBayesModel(ngram_range=ngram_range)
                    model.fit(X_train_full, y_train_full)
                    y_pred = model.predict(X_test)
                
                    elif model_type == 'naive_bayes':
                        model = NaiveBayesModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "Naive Bayes")
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                    
                    elif model_type == 'random_forest':
                        model = RandomForestModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "Random Forest")
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                    
                    elif model_type == 'xgboost':
                        model = XGBoostModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "XGBoost")
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                    
                    elif model_type == 'catboost':
                        model = CatBoostModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "CatBoost")
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                    
                    elif model_type == 'lightgbm':
                        model = LightGBMModel(ngram_range=ngram_range)
                        mlflow.log_param("model_type", "LightGBM")
                        model.fit(X_train_full, y_train_full)
                        y_pred = model.predict(X_test)
                    
                    else:
                        print(f"ERROR: Unknown model type")
                        continue
                    
                    # Ensure predictions are proper numpy arrays (avoid writebackifcopy issues)
                    y_pred = np.array(y_pred, dtype=np.int32).copy()
                    
                    # Evaluate model
                    elapsed = time.time() - start_time
                    metrics = evaluate_model(y_test, y_pred)
                    accuracy, precision, recall, f1 = metrics
                    
                    # Log metrics to MLflow
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("training_time", elapsed)
                    
                    # Save results and predictions
                    save_results(model_name, ngram_range, metrics, training_time=elapsed)
                    save_predictions(model_name, ngram_range, y_test, y_pred)
                    
                    # Display results
                    print(f"F1={f1:.4f} | Acc={accuracy:.4f} | Time={elapsed:.1f}s")
                    
                except Exception as e:
                    print(f"FAILED: {str(e)}")
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))
        
        # ============================================================================
        # RESULTS SUMMARY
        # ============================================================================
        
        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETED")
        print("="*80)
        
        # Load and display final results
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            results_df_sorted = results_df.sort_values('F1', ascending=False)
            
            print(f"\n✓ Results saved to: data/results.csv")
            print(f"✓ Total experiments: {len(results_df)}")
            
            # Display top 5 results
            print("\nTop 5 Model Configurations:")
            print("-" * 80)
            for idx, (_, row) in enumerate(results_df_sorted.head(5).iterrows(), 1):
                print(f"{idx}. {row['Model']:20s} ({row['Ngram_Range']}) - "
                      f"F1={row['F1']:.4f} | Acc={row['Accuracy']:.4f} | Time={row['Training_Time']:.1f}s")
            
            print("\nNext steps:")
            print("  1. python generate_predictions.py  # Generate confusion matrices")
            print("  2. python src/visualize.py          # Create visualizations")
        else:
            print("\n⚠ Warning: No results file found")


if __name__ == "__main__":
    DATA_CSV = "data/IMDB Dataset.csv"
    run_all_experiments(DATA_CSV)
