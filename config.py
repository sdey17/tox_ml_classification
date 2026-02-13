"""
Pipeline Configuration
======================

All pipeline settings in one place. Modify these to customize
the pipeline without touching other files.
"""


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

    # Hyperparameter optimization (Optuna)
    ENABLE_OPTUNA = True   # Set to False to skip optimization
    OPTUNA_N_TRIALS = 50   # Number of Optuna trials per model-descriptor pair
    OPTUNA_METRIC = 'ROC_AUC'  # Metric to optimize
