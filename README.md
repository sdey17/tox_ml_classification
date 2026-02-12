# Toxicity Classification Pipeline

A comprehensive machine learning pipeline for general toxicity prediction with statistical analysis and detailed visualizations. Can be adapted to any endpoints.

## Features

- **6 Molecular Descriptors**: Morgan, MACCS, RDKit, Mordred, ChemBERTa, MolFormer
- **8 ML Models**: KNN, SVM, Bayesian, Logistic Regression, Random Forest, LightGBM, XGBoost, TabPFN
- **Hyperparameter Optimization**: Optuna-based tuning per model-descriptor pair (configurable)
- **Stacking Ensemble**: Combine top-K model-descriptor pairs via stacking with a meta-learner
- **Rigorous Cross-Validation**: 5-repeat × 5-fold stratified CV (configurable)
- **Descriptor Caching**: Automatic caching of computed descriptors for faster re-runs

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Stacking Ensemble](#stacking-ensemble)
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
- Optimizes hyperparameters per model-descriptor pair using Optuna (if enabled)
- Performs 5×5 repeated stratified cross-validation with optimized hyperparameters
- Generates comprehensive performance metrics and statistical tests
- Creates visualizations (heatmaps, boxplots, comparison plots)

**Output:**
```
cv_results/
├── results_summary.csv              # Mean performance across all combinations
├── per_fold_results.csv             # Individual fold results (25 per combination)
├── optimized_hyperparameters.json   # Best hyperparameters from Optuna
├── plots/
│   ├── heatmap_ROC_AUC.png
│   ├── heatmap_MCC.png
│   ├── boxplot_Morgan_ROC_AUC.png
│   └── ...
├── statistical_tests/
│   ├── Morgan_ROC_AUC_ANOVA.txt
│   ├── Morgan_ROC_AUC_Tukey.csv
│   └── ...
└── descriptor_cache/                # Cached descriptors for reuse
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
    --hyperparams cv_results/optimized_hyperparameters.json \
    --output final_model/
```

The `--hyperparams` flag is optional. When provided, the final model is trained using the Optuna-optimized hyperparameters. When omitted, default hyperparameters are used.

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

### Step 3: Stacking Ensemble (Optional)

Instead of using a single best model, combine the top-K performers into a stacking ensemble:

```bash
python train_test_best_model.py \
    --train data/train_df.csv \
    --test data/test_df.csv \
    --results cv_results/ \
    --ensemble --top-k 5 \
    --hyperparams cv_results/optimized_hyperparameters.json \
    --output ensemble_model/
```

This selects the top 5 model-descriptor combinations by CV performance, generates out-of-fold predictions, trains a Logistic Regression meta-learner, and evaluates on the test set. It also automatically trains the single best model from CV and prints a side-by-side comparison so you can check whether the ensemble improves over the best individual model.

**Output:**
```
ensemble_model/
├── stacking_ensemble.pkl                    # Full ensemble (base models + meta-learner)
├── ensemble_metadata.json                   # Base model configs and selection info
├── ensemble_test_predictions.csv            # Ensemble per-molecule predictions
├── ensemble_test_metrics.csv                # Ensemble test set metrics
├── Morgan_XGBoost_model.pkl                 # Best single model (auto-selected)
├── Morgan_XGBoost_scaler.pkl                # Best single model scaler
├── best_model_test_predictions.csv          # Best single model predictions
├── best_model_test_metrics.csv              # Best single model metrics
├── ensemble_vs_best_model_comparison.csv    # Side-by-side metric comparison
└── descriptor_cache/                        # Cached descriptors
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
    
    # Hyperparameter optimization (Optuna)
    ENABLE_OPTUNA = True       # Set to False to skip optimization
    OPTUNA_N_TRIALS = 50       # Number of trials per model-descriptor pair
    OPTUNA_METRIC = 'ROC_AUC'  # Metric to optimize

    # Statistical tests settings
    STATS_METRICS = ['ROC_AUC', 'MCC', 'GMean']

    # Visualization settings
    VIZ_METRICS = ['ROC_AUC', 'MCC', 'GMean']
    VIZ_DPI = 300
```

## Hyperparameter Optimization

The pipeline uses [Optuna](https://optuna.org/) for automatic hyperparameter tuning. When `ENABLE_OPTUNA = True`, Optuna runs before the final cross-validation step, optimizing each model-descriptor combination independently.

**How it works:**
1. For each (descriptor, model) pair, Optuna runs `OPTUNA_N_TRIALS` trials
2. Each trial suggests hyperparameters from a predefined search space
3. Trials are evaluated using the same repeated stratified CV as the main pipeline
4. The best hyperparameters are saved to `optimized_hyperparameters.json`
5. The final CV run uses the optimized hyperparameters

**Search spaces by model:**

| Model | Tuned Hyperparameters |
|-------|----------------------|
| KNN | `n_neighbors` |
| SVM | `C`, `kernel` |
| Bayesian | `alpha` |
| LogisticRegression | `C`, `solver` |
| RandomForest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| LightGBM | `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `lambda_l1`, `lambda_l2` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight` |
| TabPFN | Skipped (pre-trained, no tunable hyperparameters) |

**To disable Optuna** and use default hyperparameters, set `ENABLE_OPTUNA = False` in the Config class.

## Stacking Ensemble

The stacking ensemble combines predictions from multiple model-descriptor pairs to improve generalization, especially when different descriptors capture complementary molecular properties.

**How it works:**
1. Select top K model-descriptor combinations from `results_summary.csv` by CV metric
2. For each base model, generate **out-of-fold (OOF) predictions** on training data using 5-fold CV
3. Train a **Logistic Regression meta-learner** on the stacked OOF predictions
4. Retrain all base models on the **full training data**
5. On the test set: each base model predicts probabilities, which are fed to the meta-learner for the final prediction

**Why this works well here:**
- Different descriptors (fingerprints vs. physicochemical vs. transformer embeddings) capture different aspects of molecular structure
- The meta-learner learns optimal weights for each base model's predictions
- OOF predictions prevent information leakage during meta-learner training

**Usage:**
```bash
# Top 5 models (default)
python train_test_best_model.py \
    --train train.csv --test test.csv \
    --results cv_results/ --ensemble --top-k 5 --output ensemble/

# Top 3 models, ranked by MCC
python train_test_best_model.py \
    --train train.csv --test test.csv \
    --results cv_results/ --ensemble --top-k 3 --metric MCC --output ensemble/
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
tox_ml_classification/
├── pipeline.py                    # Main CV pipeline (with Optuna integration)
├── train_test_best_model.py       # Train final model on test set
├── preprocess_data.py             # Data preprocessing and splitting
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── utils/
│   ├── descriptors.py             # Descriptor generation
│   ├── models.py                  # Model creation and training
│   ├── optimization.py            # Optuna hyperparameter optimization
│   ├── ensemble.py                # Stacking ensemble
│   ├── stats.py                   # Statistical tests
│   ├── plots.py                   # Visualization
│   └── preprocessing.py           # Data preprocessing utilities
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
- **SVM**: RBF kernel, balanced class weights
- **Bayesian**: Bernoulli Naive Bayes
- **LogisticRegression**: L2 regularization, balanced class weights
- **RandomForest**: 500 trees, balanced class weights
- **LightGBM**: Gradient boosting, 500 estimators
- **XGBoost**: Gradient boosting, 500 estimators, histogram tree method
- **TabPFN**: Transformer-based prior-fitted network

All model defaults can be overridden by Optuna when `ENABLE_OPTUNA = True`.

### Cross-Validation

- **Stratified K-Fold**: Preserves class distribution in each fold
- **Repeated CV**: Multiple independent runs (default: 5 repeats)
- **Consistent Seeds**: Reproducible results

### Statistical Analysis

- **RM-ANOVA**: Repeated Measures ANOVA for comparing models within each descriptor
- **Tukey HSD**: Post-hoc pairwise comparisons with family-wise error rate control

---

**Happy Modeling!**
