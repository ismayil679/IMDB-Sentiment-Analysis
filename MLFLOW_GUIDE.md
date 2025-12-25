# MLflow Integration Guide

## Overview

MLflow is now integrated into the IMDB sentiment analysis project for experiment tracking and model management.

## What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage models
- **Reproducibility**: Track code versions and dependencies
- **Comparison**: Compare different model runs

## Automatic Tracking

Every experiment run automatically logs to MLflow:

### Logged Parameters:
- Model name and type
- N-gram range
- Model-specific hyperparameters (C, max_features, solver, etc.)
- Dataset sizes (train/test samples)

### Logged Metrics:
- Accuracy
- Precision
- Recall  
- F1 Score
- Training Time

## Usage

### 1. Run Experiments (Automatically Tracked)
```bash
python src/main.py
```

All experiments are automatically tracked to `./mlruns/` directory.

### 2. View Results in MLflow UI
```bash
mlflow ui
```

Then open your browser to: `http://localhost:5000`

### 3. MLflow UI Features

**Experiments Page:**
- See all experiment runs in a table
- Sort by metrics (F1, accuracy, etc.)
- Filter runs by parameters
- Compare multiple runs side-by-side

**Run Details:**
- Full parameter set for each model
- All metrics with timestamps
- Training time comparisons
- Artifacts (if saved)

**Compare Runs:**
- Select multiple runs
- View parallel coordinates plot
- Compare metrics across runs
- Identify best configurations

### 4. Query Runs Programmatically

```python
import mlflow

# Get best run by F1 score
experiment = mlflow.get_experiment_by_name("IMDB_Sentiment_Analysis")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
best_run = runs.sort_values("metrics.f1_score", ascending=False).iloc[0]

print(f"Best Model: {best_run['params.model_name']}")
print(f"F1 Score: {best_run['metrics.f1_score']:.4f}")
```

## File Structure

```
imdb_sentiment_project/
├── mlruns/              # MLflow tracking data
│   ├── 0/               # Default experiment
│   └── .../             # Your experiments
├── mlartifacts/         # Stored artifacts (if any)
└── ...
```

## Tips

1. **Compare Configurations**: Use MLflow UI to compare different n-gram ranges, hyperparameters, etc.

2. **Track Experiments**: Each experiment run gets a unique ID for reproducibility

3. **Best Model**: Easily identify best performing configuration by sorting metrics

4. **Clean Runs**: Delete bad runs directly from MLflow UI

5. **Export Data**: Export run data to CSV for further analysis

## Example Analysis

After running experiments, use MLflow to answer questions like:
- Which model performed best? (Sort by F1 score)
- Does (1-3) n-gram improve over (1-2)? (Compare runs)
- What's the speed/accuracy tradeoff? (Plot training_time vs f1_score)
- Which hyperparameters matter most? (Parallel coordinates)

## Advanced: Model Registry

To register your best model:

```python
import mlflow

# Get the best run
best_run_id = "abc123..."  # From MLflow UI

# Register model
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "IMDB_Sentiment_Classifier")
```

## Troubleshooting

**MLflow UI not starting?**
- Check if port 5000 is available
- Try: `mlflow ui --port 5001`

**Can't see my runs?**
- Check that `mlruns/` directory exists
- Verify experiments ran successfully (no errors)

**Want to reset experiments?**
- Delete `mlruns/` directory
- Run experiments again to create fresh tracking data

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow UI](https://mlflow.org/docs/latest/tracking.html#tracking-ui)
