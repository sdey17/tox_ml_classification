"""
Main Pipeline
=============
"""

import os
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem

# Suppress common warnings
warnings.filterwarnings('ignore', message='PYTORCH_CUDA_ALLOC_CONF is deprecated')

# Import our utilities
from utils.descriptors import DescriptorGenerator, get_available_descriptors
from utils.models import CrossValidator, get_available_models
from utils.stats import StatisticalAnalyzer, create_pairwise_matrix
from utils.plots import Visualizer


# ============================================================================
# CONFIGURATION - MODIFY THIS TO CUSTOMIZE
# ============================================================================

class Config:
    """
    Pipeline configuration - all settings in one place!
    
    Modify these to customize the pipeline without touching other files.
    """
    
    # Cross-validation settings
    N_REPEATS = 5
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Descriptor settings
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 2048
    
    # Descriptor caching
    ENABLE_CACHE = True  # Set to False to disable caching
    CACHE_DIR = 'descriptor_cache'  # Directory to store cached descriptors
    
    # Which descriptors to use (comment out to skip)
    DESCRIPTORS = [
        'Morgan',
        'RDKit',
        'MACCS',
        'Mordred',
        'ChemBERTa',
        'MolFormer'
    ]
    
    # Which models to use (comment out to skip)
    MODELS = [
        'KNN',
        'SVM',
        'Bayesian',
        'LogisticRegression',
        'RandomForest',
        'LightGBM',
        'XGBoost',
        'TabPFN'
    ]
    
    # Metrics to calculate
    METRICS = [
        'ROC_AUC',
        'PR_AUC',
        'Accuracy',
        'Sensitivity',
        'Specificity',
        'GMean',
        'Precision',
        'F1',
        'MCC',
        'Kappa'
    ]
    
    # Statistical tests settings
    STATS_METRICS = ['ROC_AUC', 'MCC', 'GMean']  # Metrics to run stats on
    
    # Visualization settings
    VIZ_METRICS = ['ROC_AUC', 'MCC', 'GMean']  # Metrics to plot
    VIZ_DPI = 300


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class ToxicityPipeline:
    """
    Main pipeline orchestrator.
    
    Coordinates data loading, descriptor generation, model training,
    statistical analysis, and visualization.
    """
    
    def __init__(self, input_file, output_dir):
        """
        Initialize pipeline.
        
        Args:
            input_file: Path to CSV with SMILES and CLASS columns
            output_dir: Directory for results
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / 'plots'
        self.stats_dir = self.output_dir / 'statistical_tests'
        self.plots_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)
        
        # Setup cache directory
        if Config.ENABLE_CACHE:
            cache_dir = self.output_dir / Config.CACHE_DIR
        else:
            cache_dir = None
        
        # Initialize components
        self.desc_gen = DescriptorGenerator(
            morgan_radius=Config.MORGAN_RADIUS,
            morgan_nbits=Config.MORGAN_NBITS,
            cache_dir=cache_dir
        )
        
        self.cv = CrossValidator(
            n_repeats=Config.N_REPEATS,
            n_folds=Config.N_FOLDS,
            random_state=Config.RANDOM_STATE
        )
        
        self.stats = StatisticalAnalyzer()
        self.viz = Visualizer(self.plots_dir, dpi=Config.VIZ_DPI)
        
        print(f"\nPipeline initialized")
        print(f"Input: {input_file}")
        print(f"Output: {output_dir}")
        print(f"Device: {self.desc_gen.device}\n")
    
    def load_data(self):
        """
        Load and validate data from CSV.
        
        Returns:
            Tuple of (smiles_list, labels)
        """
        print("="*80)
        print("STEP 1: Loading Data")
        print("="*80)
        
        # Load CSV
        df = pd.read_csv(self.input_file)
        
        # Validate columns
        if 'SMILES' not in df.columns or 'CLASS' not in df.columns:
            raise ValueError("CSV must have 'SMILES' and 'CLASS' columns")
        
        # Remove missing values
        df = df.dropna(subset=['SMILES', 'CLASS']).reset_index(drop=True)
        
        # Validate SMILES
        print("Validating SMILES...")
        valid_smiles = []
        valid_labels = []
        
        for smi, label in zip(df['SMILES'], df['CLASS']):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_labels.append(label)
        
        print(f"  Total molecules: {len(df)}")
        print(f"  Valid molecules: {len(valid_smiles)}")
        print(f"  Invalid SMILES: {len(df) - len(valid_smiles)}")
        print(f"  Class distribution: {np.bincount(valid_labels)}")
        
        return valid_smiles, np.array(valid_labels)
    
    def generate_descriptors(self, smiles_list):
        """
        Generate all configured descriptors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary mapping descriptor_name -> scaled_features
        """
        print("\n" + "="*80)
        print("STEP 2: Generating Descriptors")
        print("="*80)
        
        available = get_available_descriptors()
        descriptors = {}
        
        for desc_type in Config.DESCRIPTORS:
            # Skip if not available
            if desc_type not in available:
                print(f"\nSkipping {desc_type} (not available)")
                continue
            
            try:
                print(f"\n{desc_type}:")
                
                # Generate
                X = self.desc_gen.generate(desc_type, smiles_list)
                print(f"  Shape: {X.shape}")
                
                descriptors[desc_type] = X
                
            except Exception as e:
                print(f"  Error: {e}")
        
        return descriptors
    
    def train_models(self, descriptors, y):
        """
        Train all models on all descriptors with cross-validation.
        
        Args:
            descriptors: Dictionary of descriptor_name -> features
            y: Labels
            
        Returns:
            DataFrame with per-fold results
        """
        print("\n" + "="*80)
        print("STEP 3: Training Models")
        print("="*80)
        
        available_models = get_available_models()
        all_results = []
        
        for desc_name, X in descriptors.items():
            print(f"\n{desc_name}:")
            
            for model_name in Config.MODELS:
                # Skip if not available
                if model_name not in available_models:
                    print(f"  Skipping {model_name} (not available)")
                    continue
                
                try:
                    print(f"  Training {model_name}...")
                    results = self.cv.run_cv(X, y, model_name, desc_name)
                    all_results.extend(results)
                    
                except Exception as e:
                    print(f"    Error: {e}")
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save per-fold results
        results_file = self.output_dir / 'per_fold_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved per-fold results: {results_file}")
        
        return results_df
    
    def create_summary(self, results_df):
        """
        Create summary table with mean metrics.
        
        Args:
            results_df: DataFrame with per-fold results
            
        Returns:
            DataFrame with summary statistics
        """
        print("\n" + "="*80)
        print("STEP 4: Creating Summary")
        print("="*80)
        
        # Calculate means
        summary = results_df.groupby(['Descriptor', 'Model'])[Config.METRICS].mean().reset_index()
        
        # Round
        for metric in Config.METRICS:
            summary[metric] = summary[metric].round(3)
        
        # Save
        summary_file = self.output_dir / 'results_summary.csv'
        summary.to_csv(summary_file, index=False)
        print(f"Saved summary: {summary_file}")
        
        # Show best model
        best_idx = summary['ROC_AUC'].idxmax()
        best = summary.iloc[best_idx]
        
        print(f"\nBest model (by ROC-AUC):")
        print(f"  Descriptor: {best['Descriptor']}")
        print(f"  Model: {best['Model']}")
        print(f"  ROC-AUC: {best['ROC_AUC']:.3f}")
        print(f"  MCC: {best['MCC']:.3f}")
        print(f"  GMean: {best['GMean']:.3f}")
        
        return summary
    
    def run_statistical_analysis(self, results_df):
        """
        Perform statistical tests.
        
        Args:
            results_df: DataFrame with per-fold results
        """
        print("\n" + "="*80)
        print("STEP 5: Statistical Analysis")
        print("="*80)
        
        for descriptor in results_df['Descriptor'].unique():
            print(f"\n{descriptor}:")
            
            for metric in Config.STATS_METRICS:
                # ANOVA
                anova_res = self.stats.run_anova(results_df, descriptor, metric)
                if anova_res is not None:
                    # Save ANOVA results
                    anova_file = self.stats_dir / f"{descriptor}_{metric}_ANOVA.txt"
                    with open(anova_file, 'w') as f:
                        f.write(f"Repeated Measures ANOVA: {descriptor} - {metric}\n")
                        f.write("="*60 + "\n")
                        f.write(str(anova_res))
                    
                    # Print p-value
                    p_value = anova_res.anova_table['Pr > F'].iloc[0]
                    print(f"  {metric}: p-value = {p_value:.4f}")
                
                # Tukey HSD
                tukey_res = self.stats.tukey_hsd(results_df, descriptor, metric)
                tukey_file = self.stats_dir / f"{descriptor}_{metric}_Tukey.csv"
                tukey_res.to_csv(tukey_file, index=False)
                
                # Compact Letter Display
                cld_res = self.stats.compact_letter_display(results_df, descriptor, metric)
                cld_file = self.stats_dir / f"{descriptor}_{metric}_CLD.csv"
                cld_res.to_csv(cld_file, index=False)
    
    def create_visualizations(self, results_df, summary_df):
        """
        Create all visualizations.
        
        Args:
            results_df: Per-fold results
            summary_df: Summary statistics
        """
        print("\n" + "="*80)
        print("STEP 6: Creating Visualizations")
        print("="*80)
        
        # Heatmaps
        print("\nHeatmaps:")
        for metric in Config.VIZ_METRICS:
            filename = self.viz.plot_heatmap(
                summary_df, metric,
                descriptor_order=Config.DESCRIPTORS,
                model_order=Config.MODELS
            )
            print(f"  Saved: {filename.name}")
        
        # Boxplots
        print("\nBoxplots:")
        for descriptor in results_df['Descriptor'].unique():
            for metric in ['ROC_AUC', 'MCC', 'GMean']:
                filename = self.viz.plot_boxplot(
                    results_df, descriptor, metric,
                    model_order=Config.MODELS
                )
                print(f"  Saved: {filename.name}")
        
        # P-value heatmaps
        print("\nP-value heatmaps:")
        for descriptor in results_df['Descriptor'].unique():
            for metric in ['ROC_AUC', 'MCC', 'GMean']:
                tukey_res = self.stats.tukey_hsd(results_df, descriptor, metric)
                pvalue_matrix = create_pairwise_matrix(tukey_res)
                filename = self.viz.plot_tukey_heatmap(pvalue_matrix, metric, descriptor)
                print(f"  Saved: {filename.name}")
        
        # Comparison plot
        print("\nComparison plot:")
        filename = self.viz.plot_comparison(summary_df, metrics=Config.VIZ_METRICS)
        print(f"  Saved: {filename.name}")
        
        # Combined boxplots (all descriptors together)
        print("\nCombined boxplots:")
        for metric in ['ROC_AUC', 'MCC', 'GMean']:
            filename = self.viz.create_combined_boxplot(
                results_df, metric,
                descriptor_order=Config.DESCRIPTORS,
                model_order=Config.MODELS
            )
            print(f"  Saved: {filename.name}")
        
        print(f"\nAll plots saved to: {self.plots_dir}")
    
    def run(self):
        """
        Run the complete pipeline.
        """
        start_time = datetime.now()
        
        print("\n" + "="*80)
        print("TOXICITY CLASSIFICATION PIPELINE")
        print("="*80)
        print(f"Started: {start_time:%Y-%m-%d %H:%M:%S}\n")
        print(f"Descriptors: {', '.join(Config.DESCRIPTORS)}")
        print(f"Models: {', '.join(Config.MODELS)}")
        print(f"Cross-validation: {Config.N_REPEATS}×{Config.N_FOLDS}")
        
        try:
            # Run pipeline steps
            smiles_list, y = self.load_data()
            descriptors = self.generate_descriptors(smiles_list)
            results_df = self.train_models(descriptors, y)
            summary_df = self.create_summary(results_df)
            self.run_statistical_analysis(results_df)
            self.create_visualizations(results_df, summary_df)
            
            # Done
            end_time = datetime.now()
            elapsed = end_time - start_time
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Total time: {elapsed}")
            print(f"\nResults saved to: {self.output_dir}")
            print(f"  - results_summary.csv")
            print(f"  - per_fold_results.csv")
            print(f"  - statistical_tests/ (ANOVA, Tukey HSD, CLD)")
            print(f"  - plots/ (heatmaps, boxplots, comparisons)")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: {e}")
            print(f"{'='*80}")
            raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Toxicity Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python pipeline.py --input train_df.csv --output results/
    
    # Custom output directory
    python pipeline.py --input data.csv --output my_results/

Required CSV format:
    SMILES,CLASS
    c1ccccc1,0
    ...
        """
    )
    
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input CSV file with SMILES and CLASS columns'
    )
    
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory (default: results)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = ToxicityPipeline(args.input, args.output)
    pipeline.run()


if __name__ == "__main__":
    main()
