"""
Visualization module for IMDB Sentiment Analysis Project
Generates comprehensive graphs and charts for model performance analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


class SentimentVisualizer:
    """Class to handle all visualization tasks for the sentiment analysis project"""
    
    def __init__(self, results_csv='data/results.csv', output_dir='visualizations'):
        """
        Initialize visualizer
        
        Args:
            results_csv: Path to results CSV file
            output_dir: Directory to save visualization outputs
        """
        self.results_csv = results_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        try:
            self.df = pd.read_csv(results_csv)
            print(f"✓ Loaded {len(self.df)} experiment results")
        except FileNotFoundError:
            print(f"⚠ Warning: {results_csv} not found. Please run experiments first.")
            self.df = None
    
    def plot_overall_summary(self, ngram_range=None, save=True):
        """
        Create a comprehensive overview showing all models and all metrics together
        
        Args:
            ngram_range: Filter by n-gram range ('1-2' or '1-3'). If None, uses best for each model.
        """
        if self.df is None:
            return
        
        # Filter by ngram range if specified
        if ngram_range:
            df_filtered = self.df[self.df['Ngram_Range'] == ngram_range].copy()
            title_suffix = f' (N-gram: {ngram_range})'
            filename_suffix = f'_{ngram_range.replace("-", "_")}'
        else:
            # Get best result for each model across all ngrams
            df_filtered = self.df.loc[self.df.groupby('Model')['F1'].idxmax()]
            title_suffix = ''
            filename_suffix = ''
        
        best_results = df_filtered.sort_values('F1', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create labels
        labels = best_results['Model']
        x = np.arange(len(labels))
        width = 0.2
        
        # Plot all metrics side by side
        bars1 = ax.bar(x - 1.5*width, best_results['Accuracy'], width, 
                      label='Accuracy', color='#3498db', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x - 0.5*width, best_results['Precision'], width, 
                      label='Precision', color='#2ecc71', edgecolor='black', linewidth=1.2)
        bars3 = ax.bar(x + 0.5*width, best_results['Recall'], width, 
                      label='Recall', color='#e74c3c', edgecolor='black', linewidth=1.2)
        bars4 = ax.bar(x + 1.5*width, best_results['F1'], width, 
                      label='F1-Score', color='#f39c12', edgecolor='black', linewidth=1.2)
        
        # Add value labels on F1 bars (most important metric)
        for i, (bar, val) in enumerate(zip(bars4, best_results['F1'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title(f'Overall Performance: All Models & All Metrics{title_suffix}', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.set_ylim([0.75, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add horizontal line at 0.90 (goal threshold)
        ax.axhline(y=0.90, color='green', linestyle='--', linewidth=2, 
                  alpha=0.5, label='90% Target')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f'overall_summary{filename_suffix}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
        return fig
    
    def plot_training_time_comparison(self, ngram_range=None, save=True):
        """
        Create bar chart comparing training times across all models
        Note: Requires 'Training_Time' column in results.csv
        
        Args:
            ngram_range: Filter by n-gram range ('1-2' or '1-3'). If None, uses best for each model.
        """
        if self.df is None:
            return
        
        # Check if training time data exists
        if 'Training_Time' not in self.df.columns:
            print("⚠ No training time data available. Add 'Training_Time' column to results.csv")
            return
        
        # Filter by ngram range if specified
        if ngram_range:
            df_filtered = self.df[self.df['Ngram_Range'] == ngram_range].copy()
            title_suffix = f' (N-gram: {ngram_range})'
            filename_suffix = f'_{ngram_range.replace("-", "_")}'
        else:
            # Get best result for each model (by F1)
            df_filtered = self.df.loc[self.df.groupby('Model')['F1'].idxmax()]
            title_suffix = ''
            filename_suffix = ''
        
        best_results = df_filtered.sort_values('Training_Time', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create color gradient (green for fast, red for slow)
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(best_results)))
        
        # Create bar plot
        bars = ax.bar(range(len(best_results)), 
                     best_results['Training_Time'], 
                     color=colors,
                     edgecolor='black',
                     linewidth=1.5,
                     width=0.7)
        
        # Highlight fastest model
        bars[0].set_edgecolor('green')
        bars[0].set_linewidth(3)
        
        # Add value labels on bars
        for bar, val in zip(bars, best_results['Training_Time']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add F1 scores as secondary labels
        for i, (bar, f1) in enumerate(zip(bars, best_results['F1'])):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                   f'F1: {f1:.3f}',
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_title(f'Training Time Comparison (Train + Validation Sets){title_suffix}', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(range(len(best_results)))
        ax.set_xticklabels(best_results['Model'], rotation=45, ha='right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f'training_time_comparison{filename_suffix}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
        return fig
    
    def plot_confusion_matrices(self, top_n=2, save=True):
        """
        Create confusion matrices for top N DIFFERENT models
        Note: Requires actual predictions. This method loads data from saved predictions.
        """
        if self.df is None:
            return
        
        # Check if confusion matrix data exists
        predictions_dir = Path('data/predictions')
        if not predictions_dir.exists():
            print("⚠ No prediction data available.")
            print("  To generate confusion matrices, you need to save predictions during experiments.")
            print("  Add this to your experiment code:")
            print("    np.save(f'data/predictions/{model_name}_y_test.npy', y_test)")
            print("    np.save(f'data/predictions/{model_name}_y_pred.npy', y_pred)")
            return
        
        # Get top N DIFFERENT models by F1 score
        # First, get the best configuration for each unique model
        best_per_model = self.df.loc[self.df.groupby('Model')['F1'].idxmax()]
        # Then, sort by F1 and take top N
        top_models = best_per_model.nlargest(top_n, 'F1')
        
        fig, axes = plt.subplots(1, top_n, figsize=(6*top_n, 5))
        if top_n == 1:
            axes = [axes]
        
        fig.suptitle(f'Confusion Matrices: Top {top_n} Models', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        for idx, (_, row) in enumerate(top_models.iterrows()):
            model_name = row['Model']
            ngram = row.get('Ngram_Range', '1-2')
            
            # Try to load predictions
            pred_file = predictions_dir / f"{model_name.replace(' ', '_')}_{ngram}_pred.npy"
            true_file = predictions_dir / f"{model_name.replace(' ', '_')}_{ngram}_true.npy"
            
            if pred_file.exists() and true_file.exists():
                y_true = np.load(true_file)
                y_pred = np.load(pred_file)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Plot
                ax = axes[idx]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           cbar=True, square=True, ax=ax,
                           xticklabels=['Negative', 'Positive'],
                           yticklabels=['Negative', 'Positive'])
                
                ax.set_title(f'{model_name}\nF1: {row["F1"]:.4f}', 
                           fontweight='bold', fontsize=12)
                ax.set_ylabel('True Label', fontweight='bold')
                ax.set_xlabel('Predicted Label', fontweight='bold')
                
                # Add accuracy text
                accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
                ax.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
                       ha='center', va='top', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax = axes[idx]
                ax.text(0.5, 0.5, f'Prediction data\nnot available\nfor {model_name}',
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'confusion_matrices.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
        return fig
    
    def plot_dataset_distribution(self, data_csv='data/IMDB Dataset.csv', save=True):
        """
        Visualize the dataset distribution and characteristics
        """
        try:
            df = pd.read_csv(data_csv)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('IMDB Dataset Analysis', fontsize=16, fontweight='bold')
            
            # 1. Sentiment distribution
            ax1 = axes[0, 0]
            sentiment_counts = df['sentiment'].value_counts()
            colors = ['#2ecc71' if s == 'positive' else '#e74c3c' 
                     for s in sentiment_counts.index]
            bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
                          color=colors, edgecolor='black', linewidth=1.5)
            ax1.set_title('Sentiment Distribution', fontweight='bold')
            ax1.set_xlabel('Sentiment', fontweight='bold')
            ax1.set_ylabel('Count', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontweight='bold')
            
            # 2. Review length distribution
            ax2 = axes[0, 1]
            review_lengths = df['review'].str.len()
            ax2.hist(review_lengths, bins=50, color='#3498db', 
                    edgecolor='black', alpha=0.7)
            ax2.axvline(review_lengths.median(), color='red', 
                       linestyle='--', linewidth=2, label=f'Median: {review_lengths.median():.0f}')
            ax2.set_title('Review Length Distribution (Characters)', fontweight='bold')
            ax2.set_xlabel('Length (characters)', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            # 3. Word count distribution
            ax3 = axes[1, 0]
            word_counts = df['review'].str.split().str.len()
            ax3.hist(word_counts, bins=50, color='#9b59b6', 
                    edgecolor='black', alpha=0.7)
            ax3.axvline(word_counts.median(), color='red', 
                       linestyle='--', linewidth=2, label=f'Median: {word_counts.median():.0f}')
            ax3.set_title('Review Length Distribution (Words)', fontweight='bold')
            ax3.set_xlabel('Length (words)', fontweight='bold')
            ax3.set_ylabel('Frequency', fontweight='bold')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            # 4. Dataset statistics table
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            stats_data = [
                ['Total Reviews', f'{len(df):,}'],
                ['Positive Reviews', f'{(df["sentiment"] == "positive").sum():,}'],
                ['Negative Reviews', f'{(df["sentiment"] == "negative").sum():,}'],
                ['Avg Length (chars)', f'{review_lengths.mean():.0f}'],
                ['Avg Length (words)', f'{word_counts.mean():.0f}'],
                ['Median Length (chars)', f'{review_lengths.median():.0f}'],
                ['Median Length (words)', f'{word_counts.median():.0f}'],
                ['Min Length (words)', f'{word_counts.min():.0f}'],
                ['Max Length (words)', f'{word_counts.max():.0f}']
            ]
            
            table = ax4.table(cellText=stats_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(stats_data) + 1):
                for j in range(2):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#34495e')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
                    cell.set_edgecolor('black')
                    cell.set_linewidth(1)
            
            ax4.set_title('Dataset Statistics', fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            if save:
                output_path = self.output_dir / 'dataset_distribution.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {output_path}")
            
            plt.show()
            return fig
            
        except FileNotFoundError:
            print(f"⚠ Warning: {data_csv} not found")
            return None
    
    def generate_all_visualizations(self):
        """
        Generate all essential visualizations
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        # 1. Dataset overview
        print("1. Dataset Distribution...")
        self.plot_dataset_distribution()
        
        # 2. Overall performance summary for 1-2 ngram
        print("\n2. Overall Performance Summary (1-2 N-gram)...")
        self.plot_overall_summary(ngram_range='1-2')
        
        # 3. Overall performance summary for 1-3 ngram
        print("\n3. Overall Performance Summary (1-3 N-gram)...")
        self.plot_overall_summary(ngram_range='1-3')
        
        # 4. Training time comparison for 1-2 ngram
        print("\n4. Training Time Comparison (1-2 N-gram)...")
        self.plot_training_time_comparison(ngram_range='1-2')
        
        # 5. Training time comparison for 1-3 ngram
        print("\n5. Training Time Comparison (1-3 N-gram)...")
        self.plot_training_time_comparison(ngram_range='1-3')
        
        # 6. Confusion matrices for top 2 DIFFERENT models
        print("\n6. Confusion Matrices (Top 2 Different Models)...")
        self.plot_confusion_matrices(top_n=2)
        
        print("\n" + "="*70)
        print(f"✓ All visualizations saved to: {self.output_dir.absolute()}")
        print("="*70 + "\n")
    
    def create_summary_report(self):
        """
        Create a text summary of model performance
        """
        if self.df is None:
            return
        
        report_path = self.output_dir / 'performance_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("IMDB SENTIMENT ANALYSIS - PERFORMANCE SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Best overall model
            best_idx = self.df['F1'].idxmax()
            best_model = self.df.loc[best_idx]
            
            f.write("BEST OVERALL MODEL:\n")
            f.write(f"  Model: {best_model['Model']}\n")
            f.write(f"  N-gram Range: {best_model.get('Ngram_Range', 'N/A')}\n")
            f.write(f"  Accuracy:  {best_model['Accuracy']:.4f}\n")
            f.write(f"  Precision: {best_model['Precision']:.4f}\n")
            f.write(f"  Recall:    {best_model['Recall']:.4f}\n")
            f.write(f"  F1-Score:  {best_model['F1']:.4f}\n\n")
            
            # Model ranking by F1
            f.write("MODEL RANKING (by F1-Score):\n")
            f.write("-" * 70 + "\n")
            ranked = self.df.sort_values('F1', ascending=False)
            for idx, (_, row) in enumerate(ranked.iterrows(), 1):
                f.write(f"{idx:2d}. {row['Model']:25s} "
                       f"F1: {row['F1']:.4f}  "
                       f"Acc: {row['Accuracy']:.4f}  "
                       f"N-gram: {row.get('Ngram_Range', 'N/A')}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Saved: {report_path}")


def main():
    """
    Main function to run visualizations
    """
    # Create visualizer instance
    visualizer = SentimentVisualizer()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    # Create summary report
    visualizer.create_summary_report()


if __name__ == "__main__":
    main()
