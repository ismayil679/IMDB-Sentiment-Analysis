import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def save_results(model_name, ngram_range, metrics, results_csv="data/results.csv"):
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Ngram_Range", "Accuracy", "Precision", "Recall", "F1"])

    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, f"{ngram_range[0]}-{ngram_range[1]}", 
                         f"{metrics[0]:.5f}", f"{metrics[1]:.5f}", f"{metrics[2]:.5f}", f"{metrics[3]:.5f}"])
