# IMDB Sentiment Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine-learning sentiment classifier for IMDB movie reviews. Best model (Linear SVM with (1,2) n-grams) reaches **91.58% F1** with a 54s training time and follows a validation-first evaluation process.

---

## Overview
- Focus on reproducible evaluation with a 60/20/20 stratified split and validation-only tuning
- Linear models dominate sparse TF-IDF features; SVM wins on accuracy, Logistic Regression wins on speed
- 60+ hyperparameter trials across SVM, Logistic Regression, Naive Bayes, LightGBM, XGBoost

## Results

| Model | F1 | Train Time | Note |
| --- | --- | --- | --- |
| SVM (1,2) n-grams | **91.58%** | 54s | Highest F1 |
| LogReg Elasticnet | 91.37% | 373s | Accurate but slower |
| LogReg L2-lbfgs | 91.17% | **41s** | Fastest strong model |
| SVM (1,3) n-grams | 91.50% | 123s | Slower variant |
| LightGBM | ~87–88% | ~60–90s | Tree baseline |
| XGBoost | ~86–88% | ~45–75s | Tree baseline |

Linear models (SVM, Logistic Regression) outperform tree models on high-dimensional TF-IDF.

## Quick Start

### Prerequisites
Python 3.8+, pip

### Setup
```bash
git clone https://github.com/ismayil679/IMDB-Sentiment-Analysis.git
cd imdb_sentiment_project

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### Run
```bash
python src/main.py                      # Full pipeline
python src/experiments/run_experiments.py  # Experiment sweeps
python src/visualize.py                 # Generate visualizations ⭐ NEW
```

## Visualizations 📊

The project includes a comprehensive visualization module that generates:
- **Overall Performance Summary**: All models with all 4 metrics in one comprehensive graph
- **Training Time Comparison**: Bar chart showing training time vs F1-score for all models
- **Confusion Matrices**: Detailed error analysis for top 3 models
- **Dataset Distribution**: Review length, sentiment balance, and statistics
- **Performance Summary Report**: Text-based rankings and details

All visualizations are automatically saved to the `visualizations/` directory.

```bash
# First time: Generate predictions for confusion matrices
python generate_predictions.py

# Then generate all visualizations
python src/visualize.py
```

## Project Structure

```
imdb_sentiment_project/
├── data/
│   ├── IMDB Dataset.csv              # 50,000 reviews
│   └── results.csv                   # Experiment results
├── src/
│   ├── main.py                       # Entry point
│   ├── load_data.py                  # Stratified 60/20/20 split
│   ├── preprocess.py                 # Text preprocessing pipeline
│   ├── utils.py                      # Evaluation & result saving
│   ├── visualize.py                  # Visualization module ⭐ NEW
│   ├── experiments/
│   │   └── run_experiments.py        # Hyperparameter sweeps
│   └── models/
│       ├── svm_model.py              # LinearSVC (best F1)
│       ├── logistic_regression_model.py  # LogReg variants
│       ├── naive_bayes_model.py      # Baseline
│       ├── lightgbm_model.py         # Optional
│       └── xgboost_model.py          # Optional
├── visualizations/                   # Generated graphs ⭐ NEW
│   ├── dataset_distribution.png
│   ├── overall_summary.png
│   ├── training_time_comparison.png
│   ├── confusion_matrices.png
│   └── performance_summary.txt
├── generate_predictions.py           # Generate prediction data for visualizations
├── requirements.txt
├── project_summary.txt
└── README.md
```

## Methodology
- Splits: 60% train / 20% validation / 20% test, stratified
- Hyperparameters selected only on validation; test is used once for the final report
- Minimal text preprocessing: lowercase + HTML removal; no stopword removal or lemmatization

### Best Model (Linear SVM)
```python
LinearSVC(
    C=0.4,
    max_iter=1500,
    class_weight='balanced',
    random_state=42
)

TfidfVectorizer(
    max_features=70000,
    ngram_range=(1, 2),
    min_df=2,
    dtype=np.float32,
    sublinear_tf=True,
    stop_words=None
)
```

### Performance Notes
- Multi-core where available (`n_jobs=-1`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`)
- float32 TF-IDF matrix for lower memory
- (1,2) n-grams recommended; (1,3) adds cost without gains

## Experiments
- SVM grid: C 0.3–0.6, max_features 40k–70k, max_iter 1500–2500 (48 runs)
- Logistic Regression: lbfgs vs saga; L2, L1, Elasticnet (8 runs)
- Tree baselines: LightGBM and XGBoost variants; both trail linear models by 3–4 F1 points

## Dataset
- IMDB Movie Review Dataset (Kaggle): 50k balanced reviews with `review` and `sentiment` columns
- Stored locally at `data/IMDB Dataset.csv`

## Dependencies

```
scikit-learn==1.8.0
pandas>=1.3.0
numpy>=1.21.0
nltk>=3.6.0
threadpoolctl>=3.0.0
matplotlib>=3.7.0  # ⭐ NEW
seaborn>=0.12.0    # ⭐ NEW
lightgbm>=3.3.0 (optional)
xgboost>=1.5.0 (optional)
```

## Key Insights: Precision-Recall Trade-offs

### Why Metrics Fluctuate Across Models

Different models show varying precision-recall patterns:

**Linear Models (SVM, Logistic Regression)** - Balanced Performance ✅
- Precision: ~90-91% | Recall: ~91-92%
- **Why**: Linear models excel with sparse TF-IDF features, creating smooth decision boundaries that balance both metrics
- **Best for**: Production use when balanced performance is critical

**Tree Models (Random Forest, LightGBM, XGBoost)** - High Recall, Lower Precision ⚠️
- Example: Random Forest has 81.6% precision but 89.3% recall
- **Why High Recall**: Trees create complex boundaries that capture most positive cases (few false negatives)
- **Why Low Precision**: Trees overpredict the positive class, generating many false positives
- **Root Cause**: Tree models struggle with high-dimensional sparse text features (60k+ TF-IDF dimensions)
- **Best for**: Situations where missing positives is costly (e.g., content moderation)

**Naive Bayes** - Moderate Balance
- Precision: ~84-85% | Recall: ~87-88%
- **Why**: Conditional independence assumption works reasonably well for text but isn't optimal

### Practical Implications
- **For sentiment analysis**: Use SVM or Logistic Regression for balanced, reliable predictions
- **For recall-critical tasks**: Consider Random Forest but expect more false alarms
- **For precision-critical tasks**: Linear models provide better precision without sacrificing recall

---

## Roadmap
- [x] Comprehensive visualization suite ⭐ NEW
- [x] Metric trade-off analysis ⭐ NEW
- [ ] Add k-fold cross-validation
- [ ] Evaluate transformer models (BERT, RoBERTa)
- [ ] Ensemble ideas (SVM + LogReg stacking)
- [ ] Extra features: sentiment lexicons, length, punctuation
- [ ] REST API (Flask/FastAPI)
- [ ] Advanced interpretability (LIME, SHAP)

## Contributors
- Ismayil – [@ismayil679](https://github.com/ismayil679)
- Vusat Ahmadzada – [@VusatAhmadzada](https://github.com/VusatAhmadzada)

## License
MIT License — see [LICENSE](LICENSE).

## Contact
Open an issue for questions, suggestions, or collaboration.

**Project Status**: Completed (December 2025) — 91.58% F1 achieved
