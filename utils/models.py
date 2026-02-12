"""
Model Utilities
===============

Handles all model creation, training, and evaluation.

Supported models:
- KNN
- SVM
- Bayesian (BernoulliNB)
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- TabPFN
"""

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)

import lightgbm as lgb
from xgboost import XGBClassifier

from tabpfn import TabPFNClassifier

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating ML models."""
    
    # Available model types
    AVAILABLE_MODELS = ['KNN', 'SVM', 'Bayesian', 'LogisticRegression', 'RandomForest',
                        'LightGBM', 'XGBoost', 'TabPFN']
    
    @staticmethod
    def create_knn(n_neighbors=5, **kwargs):
        """Create K-Nearest Neighbors classifier."""
        return KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

    @staticmethod
    def create_svm(kernel='rbf', random_state=42, **kwargs):
        """Create Support Vector Machine classifier."""
        return SVC(kernel=kernel, probability=True, class_weight='balanced',
                    random_state=random_state, **kwargs)

    @staticmethod
    def create_bayesian(alpha=1.0, **kwargs):
        """Create Bayesian (BernoulliNB) classifier."""
        return BernoulliNB(alpha=alpha, **kwargs)

    @staticmethod
    def create_logistic_regression(max_iter=2000, random_state=42, **kwargs):
        """Create Logistic Regression classifier."""
        return LogisticRegression(
            max_iter=max_iter,
            class_weight='balanced',
            random_state=random_state,
            **kwargs
        )

    @staticmethod
    def create_random_forest(n_estimators=500, random_state=42, **kwargs):
        """Create Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            **kwargs
        )

    @staticmethod
    def create_lightgbm(n_estimators=500, random_state=42, **kwargs):
        """Create LightGBM classifier."""
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **kwargs
        )

    @staticmethod
    def create_xgboost(n_estimators=500, random_state=42, **kwargs):
        """Create XGBoost classifier."""
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **kwargs
        )

    @staticmethod
    def create_tabpfn(device='cpu', random_state=42, **kwargs):
        """Create TabPFN classifier."""
        return TabPFNClassifier(device=device, random_state=random_state, ignore_pretraining_limits=True)
    
    @classmethod
    def create(cls, model_name, random_state=42, device='cpu', **kwargs):
        """
        Create model by name.
        
        Args:
            model_name: One of ['KNN', 'SVM', 'Bayesian', 'LogisticRegression',
                                'RandomForest', 'LightGBM', 'XGBoost', 'TabPFN']
            random_state: Random seed
            device: Device for TabPFN ('cpu' or 'cuda')
            **kwargs: Additional model-specific arguments
            
        Returns:
            Scikit-learn compatible model
        """
        factories = {
            'KNN': cls.create_knn,
            'SVM': cls.create_svm,
            'Bayesian': cls.create_bayesian,
            'LogisticRegression': cls.create_logistic_regression,
            'RandomForest': cls.create_random_forest,
            'LightGBM': cls.create_lightgbm,
            'XGBoost': cls.create_xgboost,
            'TabPFN': cls.create_tabpfn,
        }
        
        if model_name not in factories:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Pass appropriate arguments
        if model_name == 'TabPFN':
            return factories[model_name](device=device, random_state=random_state, **kwargs)
        elif model_name in ['KNN', 'Bayesian']:
            return factories[model_name](**kwargs)
        else:
            return factories[model_name](random_state=random_state, **kwargs)


# ============================================================================
# Metrics Calculator
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive classification metrics."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC-AUC and PR-AUC)
            
        Returns:
            Dictionary of metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = np.sqrt(sensitivity * specificity)
        
        metrics = {
            'ROC_AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0,
            'PR_AUC': average_precision_score(y_true, y_prob) if y_prob is not None else 0.0,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'GMean': gmean,
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics


# ============================================================================
# Cross-Validation Trainer
# ============================================================================

class CrossValidator:
    """
    Handle repeated stratified cross-validation.
    
    Usage:
        cv = CrossValidator(n_repeats=5, n_folds=5)
        results = cv.run_cv(X, y, 'XGBoost', 'Morgan')
    """
    
    def __init__(self, n_repeats=5, n_folds=5, random_state=42):
        """
        Initialize cross-validator.
        
        Args:
            n_repeats: Number of CV repeats
            n_folds: Number of folds per repeat
            random_state: Base random seed
        """
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.random_state = random_state
        self.device = DEVICE
    
    def run_cv(self, X, y, model_name, descriptor_name, model_params=None):
        """
        Run repeated stratified K-fold CV.

        Args:
            X: Feature matrix
            y: Labels
            model_name: Name of model to train
            descriptor_name: Name of descriptor (for recording)
            model_params: Optional dict of hyperparameters to pass to ModelFactory

        Returns:
            List of dictionaries with per-fold results
        """
        if model_params is None:
            model_params = {}

        results = []

        use_scaler = descriptor_name.lower() == 'mordred'
        if use_scaler:
            print(f"    Using MinMaxScaler for {descriptor_name} (fit per fold)")
        else:
            print(f"    No scaling for {descriptor_name} (already normalized)")

        total_folds = self.n_repeats * self.n_folds
        pbar = tqdm(total=total_folds, desc=f"    {model_name}", leave=False)

        for repeat in range(self.n_repeats):
            seed = self.random_state + repeat
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Scale inside the fold to prevent data leakage
                if use_scaler:
                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                # Train and evaluate
                try:
                    model = ModelFactory.create(
                        model_name, random_state=seed,
                        device=self.device, **model_params
                    )
                    model.fit(X_train, y_train)

                    # Predictions
                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

                    # Metrics
                    metrics = MetricsCalculator.calculate_metrics(y_val, y_pred, y_prob)

                    # Store
                    result = {
                        'Descriptor': descriptor_name,
                        'Model': model_name,
                        'Repeat': repeat + 1,
                        'Fold': fold + 1,
                        **metrics
                    }
                    results.append(result)

                except Exception as e:
                    print(f"      Error in fold {fold+1}: {e}")

                pbar.update(1)

        pbar.close()
        return results
    


# ============================================================================
# Convenience Functions
# ============================================================================

def get_available_models():
    """Get list of available model names."""
    return ModelFactory.AVAILABLE_MODELS


def create_model(model_name, **kwargs):
    """
    Create a model instance.
    
    Args:
        model_name: Model name
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    return ModelFactory.create(model_name, **kwargs)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    return MetricsCalculator.calculate_metrics(y_true, y_pred, y_prob)
