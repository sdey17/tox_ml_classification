# Toxicity Classification Pipeline

A comprehensive machine learning pipeline for general toxicity prediction with statistical analysis and detailed visualizations. Can be apadted to any endpoints.

## Features

- **6 Molecular Descriptors**: Morgan, MACCS, RDKit, Mordred, ChemBERTa, MolFormer
- **8 ML Models**: KNN, SVM, Bayesian, Logistic Regression, Random Forest, LightGBM, XGBoost, TabPFN
- **Rigorous Cross-Validation**: 5-repeat × 5-fold stratified CV (configurable)
- **Descriptor Caching**: Automatic caching of computed descriptors for faster re-runs

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Input Format](#input-format)
- [Output](#output)
- [Examples](#examples)

## Installation

### Prerequisites

- Python 3.11+
- CUDA (optional, for GPU acceleration with ChemBERTa, MolFormer, TabPFN)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.bhsai.net/sdey/tox_ml_classification.git
cd tox_ml_classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install other dependencies
pip install -r requirements.txt
```

**For TabPFN, accept the model license and configure the access token in Hugging Face during first time use (for details see https://docs.priorlabs.ai/how-to-access-gated-models)**

## Quick Start

### Step 0: Data Preprocessing (Optional but Recommended)

If you have raw molecular data that needs cleaning and standardization:

```bash
python preprocess_data.py --input raw_data.csv --output-dir preprocessed/
```

**What this does:**
- Validates SMILES structures using ChEMBL exclusion rules
- Standardizes and canonicalizes SMILES (removes salts, tautomers)
- Removes duplicates and conflicting labels
- Performs Butina clustering for structural diversity
- Creates train/test split maintaining cluster integrity
- Analyzes train/test similarity with Tanimoto similarity
- Generates visualization of similarity distribution between training and test sets

### Step 1: Cross-Validation Pipeline

Run the full pipeline to identify the best model/descriptor combination:

```bash
python pipeline.py --input data/train_df.csv --output cv_results/
```

**What this does:**
- Tests all descriptor types (Morgan, Mordred, ChemBERTa, MolFormer, etc.)
- Tests all models (KNN, SVM, XGBoost, LightGBM, RandomForest, etc.)
- Performs 5×5 repeated stratified cross-validation
- Generates comprehensive performance metrics and statistical tests
- Creates visualizations (heatmaps, boxplots, comparison plots)

**Output:**
```
cv_results/
├── results_summary.csv          # Mean performance across all combinations
├── per_fold_results.csv         # Individual fold results (25 per combination)
├── plots/
│   ├── heatmap_ROC_AUC.png
│   ├── heatmap_MCC.png
│   ├── boxplot_Morgan_ROC_AUC.png
│   └── ...
├── statistical_tests/
│   ├── Morgan_ROC_AUC_ANOVA.txt
│   ├── Morgan_ROC_AUC_Tukey.csv
│   └── ...
└── descriptor_cache/            # Cached descriptors for reuse
```
### Step 2: Train Final Model

After cross-validation identifies the best model, train on full training data and evaluate on test set:

```bash
# Auto-select best model from CV results using ROC_AUC (recommended)
python train_test_best_model.py \
    --train data/train_df.csv \
    --test data/test_df.csv \
    --results cv_results/ \
    --metric ROC_AUC \
    --output final_model/
```

**Output:**
```
final_model/
├── Morgan_XGBoost_model.pkl     # Trained model
├── Morgan_XGBoost_scaler.pkl    # Fitted scaler (None for non-Mordred)
├── model_metadata.json          # Model configuration and info
├── test_predictions.csv         # Per-molecule predictions
├── test_metrics.csv             # Test set metrics
└── descriptor_cache/            # Cached descriptors
```

## Configuration

### Edit `Config` Class in `pipeline.py`

All pipeline settings are in the `Config` class at the top of `pipeline.py`. Modify these to customize the pipeline without touching other files.

```python
class Config:
    """Pipeline configuration - all settings in one place!"""
    
    # Cross-validation settings
    N_REPEATS = 5          # Number of CV repeats
    N_FOLDS = 5            # Number of CV folds
    RANDOM_STATE = 42      # Random seed for reproducibility
    
    # Descriptor settings
    MORGAN_RADIUS = 2      # Morgan fingerprint radius
    MORGAN_NBITS = 2048    # Morgan fingerprint bits
    
    # Descriptor caching
    ENABLE_CACHE = True    # Set to False to disable caching
    CACHE_DIR = 'descriptor_cache'
    
    # Which descriptors to use (comment out to skip)
    DESCRIPTORS = [
        'Morgan',
        'RDKit',
        'MACCS',
        'Mordred',       # Requires mordred package
        'ChemBERTa',     # Requires transformers + GPU
        'MolFormer'      # Requires transformers + GPU
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
        'TabPFN'         # Requires GPU
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
    STATS_METRICS = ['ROC_AUC', 'MCC', 'GMean']
    
    # Visualization settings
    VIZ_METRICS = ['ROC_AUC', 'MCC', 'GMean']
    VIZ_DPI = 300
```

## Input Format

Input CSV must contain:
- `SMILES`: SMILES strings of molecules
- `CLASS`: Binary labels (0 = non-toxic, 1 = toxic)

Example `train_df.csv`:
```csv
SMILES,CLASS
CCO,0
c1ccccc1,0
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,1
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,1
```

**Notes:**
- Column names must be exactly `SMILES` and `CLASS` (case-sensitive)
- Invalid SMILES will be automatically filtered out
- Class distribution will be reported during data loading


### Performance Metrics

All metrics are calculated automatically:

- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **MCC**: Matthews Correlation Coefficient
- **GMean**: Geometric mean of sensitivity and specificity
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Precision**, **F1**, **Accuracy**, **Kappa**

## Examples

### Example 1: Quick Test Run

```python
# In pipeline.py Config class:
DESCRIPTORS = ['Morgan', 'MACCS']
MODELS = ['LogisticRegression', 'RandomForest']
N_REPEATS = 3
N_FOLDS = 3
```

```bash
python pipeline.py --input data/test_df.csv --output quick_test/
```

### Example 2: Comprehensive Analysis

```python
# In pipeline.py Config class:
DESCRIPTORS = ['Morgan', 'RDKit', 'MACCS', 'Mordred', 'ChemBERTa', 'MolFormer']
MODELS = ['KNN', 'SVM', 'Bayesian', 'LogisticRegression', 'RandomForest', 'LightGBM', 'XGBoost', 'TabPFN']
N_REPEATS = 5
N_FOLDS = 5
```

```bash
python pipeline.py --input data/train_df.csv --output full_analysis/
```

### Example 3: Re-run with Cached Descriptors

After running the pipeline once, descriptors are cached. Modify Config to use different models:

```python
# Keep descriptors the same:
DESCRIPTORS = ['Morgan', 'MACCS', 'RDKit']

# Change models:
MODELS = ['SVM', 'KNN', 'Bayesian']  # Different models!

# Cache is enabled:
ENABLE_CACHE = True  # Will reuse cached descriptors!
```

```bash
# This will be much faster - descriptors loaded from cache
python pipeline.py --input data/train_df.csv --output new_models_results/
```

## Repository Structure

```
toxicity-pipeline/
├── pipeline.py                    # Main CV pipeline
├── train_test_best_model.py       # Train final model
├── utils_descriptors.py           # Descriptor generation
├── utils_models.py                # Model creation and training
├── utils_stats.py                 # Statistical tests
├── utils_plots.py                 # Visualization
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── data/
    ├── train_df.csv
    └── test_df.csv
```

## Methodology

### Molecular Descriptors

- **Morgan**: Circular fingerprints (similar to ECFP)
- **MACCS**: 166-bit MACCS keys
- **RDKit**: RDKit topological fingerprints
- **Mordred**: 1613 2D/3D molecular descriptors
- **ChemBERTa**: Transformer-based molecular embeddings
- **MolFormer**: Transformer-based molecular representations

### Machine Learning Models

- **KNN**: k=5 neighbors
- **SVM**: RBF kernel with class weights
- **Bayesian**: Bernoulli Naive Bayes
- **LogisticRegression**: L2 regularization, class weights
- **RandomForest**: 100 trees, class weights
- **LightGBM**: Gradient boosting
- **XGBoost**: Gradient boosting with scale_pos_weight
- **TabPFN**: Transformer-based prior-fitted network

### Cross-Validation

- **Stratified K-Fold**: Preserves class distribution in each fold
- **Repeated CV**: Multiple independent runs (default: 5 repeats)
- **Consistent Seeds**: Reproducible results

### Statistical Analysis

- **RM-ANOVA**: Repeated Measures ANOVA for comparing models within each descriptor
- **Tukey HSD**: Post-hoc pairwise comparisons with family-wise error rate control

---

**Happy Modeling!**
