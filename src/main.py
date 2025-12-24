"""Main entry point for IMDB sentiment analysis experiments.

Simple wrapper that invokes the full experimental pipeline.
Runs all model experiments with proper train/val/test evaluation.

Usage:
    python src/main.py
    
For more control, run experiments directly:
    python src/experiments/run_experiments.py
"""

from experiments.run_experiments import run_all_experiments


if __name__ == "__main__":
    # Path to IMDB dataset CSV (50,000 reviews)
    DATA_CSV = "data/IMDB Dataset.csv"
    
    # Run all model experiments
    run_all_experiments(DATA_CSV)
