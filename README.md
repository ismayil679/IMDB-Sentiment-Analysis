# IMDB Sentiment Analysis Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance sentiment classification system for IMDB movie reviews achieving **91.58% F1 score** through optimized machine learning techniques and proper validation methodology.

---

## 🎯 Project Highlights

- **91.58% F1 Score**: Exceeds 90% accuracy goal with Support Vector Machine (SVM)
- **Fast Training**: 54-second training time with multi-core optimizations
- **Proper ML Methodology**: Validation-based hyperparameter tuning prevents test set leakage
- **Comprehensive Comparison**: Evaluated 6 models with 60+ hyperparameter configurations
- **Production-Ready**: Optimized for both accuracy and inference speed

---

## 📊 Results Summary

| Model | F1 Score | Training Time | Status |
|-------|----------|---------------|--------|
| **SVM (1,2) n-grams** | **91.58%** | 54s | 🏆 Best Overall |
| LogReg Elasticnet | 91.37% | 373s | High Accuracy |
| **LogReg L2-lbfgs** | **91.17%** | **41s** | ⚡ Best Speed |
| SVM (1,3) n-grams | 91.50% | 123s | Slower |
| LightGBM | ~87-88% | ~60-90s | Tree Model |
| XGBoost | ~86-88% | ~45-75s | Tree Model |

**Key Finding**: Linear models (SVM, Logistic Regression) significantly outperform tree-based models (LightGBM, XGBoost) on high-dimensional sparse TF-IDF features.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ismayil679/IMDB-Sentiment-Analysis.git
cd imdb_sentiment_project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Run main experiment pipeline
python src/main.py

# Run specific experiments (after configuration)
python src/experiments/run_experiments.py
```

---

## 📁 Project Structure

```
imdb_sentiment_project/
├── data/
│   └── IMDB Dataset.csv              # 50,000 movie reviews
├── src/
│   ├── main.py                       # Entry point
│   ├── load_data.py                  # Stratified 60/20/20 split
│   ├── preprocess.py                 # Text preprocessing pipeline
│   ├── utils.py                      # Evaluation & results saving
│   ├── experiments/
│   │   └── run_experiments.py        # Hyperparameter tuning & testing
│   └── models/
│       ├── svm_model.py              # LinearSVC (Winner: 91.58% F1)
│       ├── logistic_regression_model.py  # LogReg variants
│       ├── naive_bayes_model.py      # Baseline model
│       ├── lightgbm_model.py         # Gradient boosting (optional)
│       └── xgboost_model.py          # Gradient boosting (optional)
├── requirements.txt                   # Project dependencies
├── project_summary.txt                # Detailed methodology & results
└── README.md                          # This file
```

---

## 🧪 Methodology

### Data Splitting Strategy
- **60% Training** (30,000 samples): Model training
- **20% Validation** (10,000 samples): Hyperparameter tuning
- **20% Test** (10,000 samples): Final evaluation (used once)

All splits are **stratified** to maintain balanced class distribution.

### Validation-Based Tuning
- All hyperparameter selection performed on validation set
- Test set reserved for final evaluation only
- Prevents test set leakage and ensures honest performance estimates

### Best Model Configuration (SVM)
```python
LinearSVC(
    C=0.4,                    # Optimal regularization
    max_iter=1500,            # Sufficient convergence
    class_weight='balanced',
    random_state=42
)

TfidfVectorizer(
    max_features=70000,       # High-dimensional feature space
    ngram_range=(1, 2),       # Unigrams + bigrams
    min_df=2,                 # Filter rare tokens
    dtype=np.float32,         # Memory optimization
    sublinear_tf=True,        # Improved TF scaling
    stop_words=None           # Keep all words
)
```

### Performance Optimizations
- **Multi-core Threading**: 8-core CPU utilization (`n_jobs=-1`)
- **Environment Variables**: `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`
- **Memory Efficiency**: float32 dtype reduces memory by 50%
- **Feature Engineering**: Optimal TF-IDF parameters found through validation

---

## 🔬 Experiments Conducted

### 1. SVM Hyperparameter Grid Search
- **Configurations**: 48 validation runs
- **Parameters Tuned**: C [0.3-0.6], max_features [40k-70k], max_iter [1500-2500]
- **Best Result**: C=0.4, max_features=70k, max_iter=1500 → **91.58% F1**

### 2. Logistic Regression Solver Comparison
- **Configurations**: 8 validation runs
- **Solvers Tested**: lbfgs, saga
- **Penalties Tested**: L2, L1, Elasticnet
- **Key Finding**: L2+lbfgs is 7× faster than Elasticnet+saga with only 0.2% F1 drop

### 3. Tree Model Evaluation
- **LightGBM**: Tested 7 configurations (n_estimators, max_depth, learning_rate, num_leaves)
- **XGBoost**: Tested 8 configurations (n_estimators, max_depth, learning_rate, subsample)
- **Conclusion**: Tree models 3-4% behind linear models due to sparse feature space

### 4. N-gram Comparison
- **(1,2) n-grams**: 91.58% F1, 54s training
- **(1,3) n-grams**: 91.50% F1, 123s training (2.5× slower)
- **Recommendation**: Use (1,2) n-grams - better speed/accuracy tradeoff

---

## 📈 Key Insights

### Why Linear Models Win
1. **High-Dimensional Sparse Data**: TF-IDF creates 70,000 sparse features
2. **Linear Separability**: Sentiment is often linearly separable in TF-IDF space
3. **Tree Model Limitations**: Decision trees need dense features to split effectively
4. **Computational Efficiency**: Linear models scale better with feature dimensionality

### Preprocessing Choices
- **Minimal Preprocessing**: Lowercase + HTML removal only
- **Stopwords**: Kept (removal didn't improve accuracy)
- **Lemmatization**: Skipped (too slow without accuracy gain)
- **Rationale**: Let TF-IDF handle feature importance weighting

### Production Recommendations
- **For Maximum Accuracy**: Use SVM with (1,2) n-grams (91.58% F1)
- **For Interpretability**: Use LogReg L2-lbfgs (91.17% F1, 41s)
- **For Speed**: LogReg L2-lbfgs is fastest while maintaining competitive accuracy
- **Avoid**: Tree models and (1,3) n-grams for this dataset

---

## 📦 Dependencies

```
scikit-learn==1.8.0
pandas>=1.3.0
numpy>=1.21.0
nltk>=3.6.0
threadpoolctl>=3.0.0
lightgbm>=3.3.0 (optional)
xgboost>=1.5.0 (optional)
```

---

## 🧮 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive identification rate
- **F1-Score**: Harmonic mean of precision and recall (primary metric)

All metrics tracked with training time measurements for each configuration.

---

## 📚 Dataset

**IMDB Movie Review Dataset**
- **Source**: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 reviews
- **Balance**: 25,000 positive / 25,000 negative
- **Format**: CSV with 'review' and 'sentiment' columns

---

## 🛠️ Future Improvements

- [ ] Implement k-fold cross-validation for robust estimates
- [ ] Test transformer models (BERT, RoBERTa) for potential gains
- [ ] Add ensemble methods (SVM + LogReg stacking)
- [ ] Feature engineering: sentiment lexicons, review length, punctuation
- [ ] Deploy as REST API with Flask/FastAPI
- [ ] Add model interpretation visualizations (LIME, SHAP)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ismayil**
- GitHub: [@ismayil679](https://github.com/ismayil679)
- Repository: [IMDB-Sentiment-Analysis](https://github.com/ismayil679/IMDB-Sentiment-Analysis)

---

## 🙏 Acknowledgments

- IMDB Dataset providers
- scikit-learn development team
- Open-source ML community

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the GitHub repository.

---

**Project Status**: ✅ Completed (December 2025)  
**Achievement**: 🏆 91.58% F1 Score - Exceeding 90% Goal
