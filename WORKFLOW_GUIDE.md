# Project Workflow Guide

## How to Run Complete Analysis with Automatic Updates

### ✅ **YES! Everything Updates Automatically**

When you run experiments again, the following happens automatically:

1. **`results.csv` is cleared** and rebuilt from scratch
2. **Training times are saved** for each model
3. **Predictions are saved** to `data/predictions/` for confusion matrices
4. **Visualizations update** based on new results.csv data

---

## Complete Workflow

### **Step 1: Run Experiments**
```bash
python src/main.py
# OR
python src/experiments/run_experiments.py
```

**What happens:**
- Clears old `data/results.csv`
- Trains all models with timing
- Saves results with 7 columns: Model, Ngram_Range, Accuracy, Precision, Recall, F1, Training_Time
- Saves predictions automatically via `save_predictions()` function
- Output: Updated `data/results.csv` with new metrics

### **Step 2: Generate Predictions (for Confusion Matrices)**
```bash
python generate_predictions.py
```

**What happens:**
- Trains top 2 models (SVM and Logistic Regression)
- Saves true labels and predictions to `data/predictions/`
- Files created: `SVM_1-2_true.npy`, `SVM_1-2_pred.npy`, `Logistic_Regression_1-2_true.npy`, `Logistic_Regression_1-2_pred.npy`
- Only needed once or when you want to update confusion matrices

### **Step 3: Generate Visualizations**
```bash
python src/visualize.py
```

**What happens:**
- Reads current `data/results.csv`
- Generates 5 visualization files:
  1. `dataset_distribution.png` - Dataset overview
  2. `overall_summary.png` - All models, all metrics
  3. `training_time_comparison.png` - Speed vs accuracy
  4. `confusion_matrices.png` - Error analysis (top 2 models: SVM & LogReg)
  5. `performance_summary.txt` - Text rankings
- **Automatically reflects latest results.csv data**

---

## What Updates Automatically?

| Component | Updates When You Run Experiments? | Source |
|-----------|----------------------------------|--------|
| `results.csv` | ✅ YES - Rebuilt from scratch | `run_experiments.py` |
| Training times | ✅ YES - Auto-saved per model | `save_results(..., training_time=elapsed)` |
| Predictions | ✅ YES - If `save_predictions()` called | `save_predictions(...)` in experiments |
| Visualizations | ✅ YES - Reads latest results.csv | `visualize.py` |
| Confusion matrices | ⚠️ REQUIRES predictions in `data/predictions/` | `generate_predictions.py` |

---

## Modified Functions

### **`utils.py`**
```python
# Now includes training_time parameter
def save_results(model_name, ngram_range, metrics, training_time=None, results_csv="data/results.csv")

# New function to save predictions for confusion matrices
def save_predictions(model_name, ngram_range, y_true, y_pred, predictions_dir="data/predictions")
```

### **`run_experiments.py`**
```python
# Now saves time and predictions
elapsed = time.time() - start_time
metrics = evaluate_model(y_test, y_pred)
save_results("ModelName", ngram, metrics, training_time=elapsed)  # ← training_time added
save_predictions("ModelName", ngram, y_test, y_pred)  # ← predictions saved
```

---

## XGBoost Issue

**Problem:** XGBoost installation conflicts with Python 3.13  
**Status:** Currently not available in this environment  
**Workaround:** Visualizations work with available models (SVM, LogReg, etc.)  
**Solution:** Install in a compatible environment or use conda:
```bash
conda install -c conda-forge xgboost
```

---

## Quick Reference

### Full Pipeline (One Command Each)
```bash
# 1. Run all experiments (updates results.csv automatically)
python src/main.py

# 2. Generate predictions for confusion matrices
python generate_predictions.py

# 3. Create all visualizations
python src/visualize.py
```

### Check Current Results
```bash
# View results table
python -c "import pandas as pd; print(pd.read_csv('data/results.csv'))"

# Check if predictions exist
ls data/predictions/
```

---

## File Locations

### Input Data
- `data/IMDB Dataset.csv` - Original dataset (50k reviews)

### Output Files
- `data/results.csv` - **Auto-updated** metrics for all models
- `data/predictions/*.npy` - Saved predictions for confusion matrices
- `visualizations/*.png` - **Auto-updated** graphs
- `visualizations/performance_summary.txt` - **Auto-updated** text report

---

## Summary

✅ **YES, everything updates automatically when you re-run experiments!**

The workflow is designed so that:
1. Running `python src/main.py` updates all metrics including times
2. Running `python src/visualize.py` always uses the latest data
3. No manual editing of CSV files needed
4. Visualizations dynamically adapt to whatever models are in results.csv

**Just run the three commands above and everything stays in sync!**
