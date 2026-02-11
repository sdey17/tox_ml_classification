#!/usr/bin/env python
"""
Molecular Data Preprocessing Script
====================================

Prepares molecular toxicity datasets for machine learning:
1. Validates SMILES structures
2. Standardizes and canonicalizes using ChEMBL pipeline
3. Removes duplicates and conflicting labels
4. Performs Butina clustering for structural diversity
5. Creates train/test split maintaining cluster integrity
6. Analyzes train/test similarity with Tanimoto coefficients

Usage:
    python preprocess_data.py --input raw_data.csv --output-dir preprocessed/
    
    python preprocess_data.py \\
        --input raw_data.csv \\
        --output-dir preprocessed/ \\
        --test-size 0.25 \\
        --no-butina
"""

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocessing import (
    preprocess_dataset,
    calculate_train_test_similarity,
    get_preprocessing_statistics,
)

warnings.filterwarnings('ignore')


# ============================================================================
# Visualization
# ============================================================================

def plot_similarity_distribution(similarities: np.ndarray, 
                                  output_file: Path,
                                  title: str = "Train-Test Tanimoto Similarity"):
    """
    Create histogram of Tanimoto similarities.
    
    Args:
        similarities: Array of similarity values
        output_file: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.histplot(similarities, bins=40, color='red', kde=True)
    ax.set_xlabel("Tanimoto Similarity (Morgan FPs)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(labelsize=12)
    
    # Add statistics
    mean_sim = similarities.mean()
    median_sim = np.median(similarities)
    ax.axvline(mean_sim, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_sim:.3f}')
    ax.axvline(median_sim, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median_sim:.3f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved similarity plot: {output_file.name}")


def plot_class_distribution(train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             output_file: Path,
                             label_col: str = 'CLASS'):
    """
    Create bar plot comparing class distributions.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        output_file: Path to save figure
        label_col: Name of label column
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train distribution
    train_counts = train_df[label_col].value_counts().sort_index()
    axes[0].bar(train_counts.index, train_counts.values, color='steelblue')
    axes[0].set_xlabel('Class', fontsize=14)
    axes[0].set_ylabel('Count', fontsize=14)
    axes[0].set_title(f'Training Set (n={len(train_df)})', fontsize=16)
    axes[0].tick_params(labelsize=12)
    
    # Test distribution
    test_counts = test_df[label_col].value_counts().sort_index()
    axes[1].bar(test_counts.index, test_counts.values, color='coral')
    axes[1].set_xlabel('Class', fontsize=14)
    axes[1].set_ylabel('Count', fontsize=14)
    axes[1].set_title(f'Test Set (n={len(test_df)})', fontsize=16)
    axes[1].tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved class distribution: {output_file.name}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess molecular toxicity data for ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic preprocessing with Butina clustering
    python preprocess_data.py --input raw_data.csv --output-dir preprocessed/
    
    # Custom test size and random seed
    python preprocess_data.py \\
        --input raw_data.csv \\
        --output-dir preprocessed/ \\
        --test-size 0.25 \\
        --random-state 123
    
    # Skip Butina clustering (faster, less rigorous)
    python preprocess_data.py \\
        --input raw_data.csv \\
        --output-dir preprocessed/ \\
        --no-butina
    
    # Skip validation and standardization (if already done)
    python preprocess_data.py \\
        --input raw_data.csv \\
        --output-dir preprocessed/ \\
        --no-validate \\
        --no-standardize

Input CSV format:
    SMILES,CLASS
    CCO,0
    CC(=O)O,1
    ...

Output files:
    preprocessed/
    ├── train_df.csv              # Training data
    ├── test_df.csv               # Test data
    ├── preprocessing_stats.json  # Statistics
    ├── preprocessing_stats.csv   # Statistics (readable)
    ├── similarity_distribution.png
    └── class_distribution.png
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file with SMILES and CLASS columns')
    parser.add_argument('--output-dir', '-o', default='preprocessed',
                        help='Output directory for processed files (default: preprocessed)')
    
    # Optional parameters
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Column names
    parser.add_argument('--smiles-col', default='SMILES',
                        help='Name of SMILES column (default: SMILES)')
    parser.add_argument('--label-col', default='CLASS',
                        help='Name of label column (default: CLASS)')
    
    # Pipeline options
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip structure validation')
    parser.add_argument('--no-standardize', action='store_true',
                        help='Skip SMILES standardization')
    parser.add_argument('--no-butina', action='store_true',
                        help='Skip Butina clustering (use random split)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("\nMOLECULAR DATA PREPROCESSING")
    print(f"Started: {start_time:%Y-%m-%d %H:%M:%S}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    
    # Dependencies are guaranteed by requirements.txt

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"\nLoading data from {args.input}...")
        df = pd.read_csv(args.input)
        
        # Validate columns
        if args.smiles_col not in df.columns:
            raise ValueError(f"Column '{args.smiles_col}' not found. Available: {df.columns.tolist()}")
        if args.label_col not in df.columns:
            raise ValueError(f"Column '{args.label_col}' not found. Available: {df.columns.tolist()}")
        
        print(f"  Loaded {len(df)} molecules")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Run preprocessing
        train_df, test_df, stats = preprocess_dataset(
            df,
            test_size=args.test_size,
            smiles_col=args.smiles_col,
            label_col=args.label_col,
            validate=not args.no_validate,
            standardize=not args.no_standardize,
            use_butina=not args.no_butina,
            random_state=args.random_state
        )
        
        # Save datasets
        print("\nSAVING RESULTS")
        
        train_file = output_dir / 'train_df.csv'
        test_file = output_dir / 'test_df.csv'
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"\nSaved training data: {train_file}")
        print(f"  Shape: {train_df.shape}")
        print(f"Saved test data: {test_file}")
        print(f"  Shape: {test_df.shape}")
        
        # Save statistics
        stats_json = output_dir / 'preprocessing_stats.json'
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved statistics: {stats_json.name}")
        
        stats_df = get_preprocessing_statistics(stats)
        stats_csv = output_dir / 'preprocessing_stats.csv'
        stats_df.to_csv(stats_csv)
        print(f"Saved statistics table: {stats_csv.name}")
        
        # Generate visualizations
        if not args.no_plots:
            print("\nGENERATING VISUALIZATIONS")
            
            # Similarity distribution
            similarities = calculate_train_test_similarity(
                train_df[args.smiles_col].tolist(),
                test_df[args.smiles_col].tolist(),
                top_n=5
            )
            
            sim_plot = output_dir / 'similarity_distribution.png'
            plot_similarity_distribution(similarities, sim_plot)
            
            # Class distribution
            class_plot = output_dir / 'class_distribution.png'
            plot_class_distribution(train_df, test_df, class_plot, args.label_col)
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"Duration: {duration}")
        print(f"\nOutput directory: {args.output_dir}")
        print(f"  Files created:")
        print(f"    - train_df.csv ({len(train_df)} molecules)")
        print(f"    - test_df.csv ({len(test_df)} molecules)")
        print(f"    - preprocessing_stats.json")
        print(f"    - preprocessing_stats.csv")
        if not args.no_plots:
            print(f"    - similarity_distribution.png")
            print(f"    - class_distribution.png")
        
        print(f"\nNext step:")
        print(f"  python pipeline.py --input {train_file} --output cv_results/")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
