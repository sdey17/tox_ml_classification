"""
Stacking Ensemble
=================

Stacking ensemble that combines top-K model-descriptor combinations
from cross-validation results. Uses out-of-fold predictions as
meta-features for a Logistic Regression meta-learner.

Flow:
    1. Select top K base models by CV metric from results_summary.csv
    2. Generate out-of-fold (OOF) predictions on training data via CV
    3. Train meta-learner on OOF predictions
    4. Retrain all base models on full training data
    5. Predict on test set: base models -> meta-learner -> final prediction
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from utils.models import ModelFactory, MetricsCalculator


def select_top_k(results_dir, k=5, metric='ROC_AUC'):
    """
    Select top K model-descriptor combinations from CV results.

    Args:
        results_dir: Path to CV results directory
        k: Number of top combinations to select
        metric: Metric to rank by

    Returns:
        List of (descriptor, model, score) tuples
    """
    results_file = Path(results_dir) / 'results_summary.csv'
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            "Run cross-validation first with pipeline.py"
        )

    df = pd.read_csv(results_file)
    df = df.sort_values(metric, ascending=False).head(k)

    selections = []
    for _, row in df.iterrows():
        selections.append((row['Descriptor'], row['Model'], row[metric]))

    return selections


class StackingEnsemble:
    """
    Stacking ensemble using out-of-fold predictions.

    Base models produce probability predictions that become features
    for a Logistic Regression meta-learner.
    """

    def __init__(self, base_configs, desc_generator, hyperparams=None,
                 n_folds=5, random_state=42):
        """
        Args:
            base_configs: List of (descriptor_name, model_name) tuples
            desc_generator: DescriptorGenerator instance
            hyperparams: Dict mapping "descriptor_model" -> params (optional)
            n_folds: Folds for generating OOF predictions
            random_state: Random seed
        """
        self.base_configs = base_configs
        self.desc_gen = desc_generator
        self.hyperparams = hyperparams or {}
        self.n_folds = n_folds
        self.random_state = random_state

        self.meta_learner = LogisticRegression(
            max_iter=2000, random_state=random_state
        )
        self.base_models = []   # Fitted base models (on full training data)
        self.scalers = []       # Fitted scalers per base model
        self.descriptors_train = {}  # Cached descriptor matrices
        self.descriptors_test = {}

    def _get_params(self, desc_name, model_name):
        key = f"{desc_name}_{model_name}"
        return self.hyperparams.get(key, {})

    def _generate_descriptors(self, desc_name, train_smiles, test_smiles):
        """Generate and cache descriptors for a descriptor type."""
        if desc_name not in self.descriptors_train:
            all_smiles = train_smiles + test_smiles
            X_all = self.desc_gen.generate(desc_name, all_smiles)
            n_train = len(train_smiles)
            self.descriptors_train[desc_name] = X_all[:n_train]
            self.descriptors_test[desc_name] = X_all[n_train:]

        return self.descriptors_train[desc_name], self.descriptors_test[desc_name]

    def fit(self, train_smiles, y_train, test_smiles):
        """
        Fit the stacking ensemble.

        1. Generate OOF predictions for meta-learner training
        2. Train meta-learner on OOF predictions
        3. Retrain base models on full training data

        Args:
            train_smiles: List of training SMILES
            y_train: Training labels
            test_smiles: List of test SMILES (for descriptor generation)
        """
        n_base = len(self.base_configs)
        n_train = len(y_train)

        # OOF predictions matrix: each column is one base model's probabilities
        oof_predictions = np.zeros((n_train, n_base))

        print(f"\nGenerating out-of-fold predictions ({self.n_folds} folds)...")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True,
            random_state=self.random_state
        )

        for i, (desc_name, model_name) in enumerate(self.base_configs):
            print(f"\n  [{i+1}/{n_base}] {desc_name} + {model_name}")

            X_train, X_test = self._generate_descriptors(
                desc_name, train_smiles, test_smiles
            )
            params = self._get_params(desc_name, model_name)
            use_scaler = desc_name.lower() == 'mordred'

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr = y_train[tr_idx]

                if use_scaler:
                    scaler = MinMaxScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_val = scaler.transform(X_val)

                model = ModelFactory.create(
                    model_name, random_state=self.random_state, **params
                )
                model.fit(X_tr, y_tr)

                if hasattr(model, 'predict_proba'):
                    oof_predictions[val_idx, i] = model.predict_proba(X_val)[:, 1]
                else:
                    oof_predictions[val_idx, i] = model.predict(X_val).astype(float)

            print(f"    OOF mean probability: {oof_predictions[:, i].mean():.4f}")

        # Train meta-learner on OOF predictions
        print("\nTraining meta-learner (Logistic Regression) on OOF predictions...")
        self.meta_learner.fit(oof_predictions, y_train)
        print("  Meta-learner trained")

        # Retrain all base models on full training data
        print("\nRetraining base models on full training data...")
        self.base_models = []
        self.scalers = []

        for i, (desc_name, model_name) in enumerate(self.base_configs):
            X_train = self.descriptors_train[desc_name]
            params = self._get_params(desc_name, model_name)
            use_scaler = desc_name.lower() == 'mordred'

            if use_scaler:
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
            else:
                scaler = None
                X_train_scaled = X_train

            model = ModelFactory.create(
                model_name, random_state=self.random_state, **params
            )
            model.fit(X_train_scaled, y_train)

            self.base_models.append(model)
            self.scalers.append(scaler)
            print(f"  [{i+1}/{n_base}] {desc_name} + {model_name}")

        return oof_predictions

    def predict_proba(self, test_smiles=None):
        """
        Generate stacked predictions on test data.

        Returns:
            Array of predicted probabilities (class 1)
        """
        n_base = len(self.base_configs)
        # Use first test set to determine size
        first_desc = self.base_configs[0][0]
        n_test = self.descriptors_test[first_desc].shape[0]

        test_predictions = np.zeros((n_test, n_base))

        for i, (desc_name, model_name) in enumerate(self.base_configs):
            X_test = self.descriptors_test[desc_name]
            scaler = self.scalers[i]
            model = self.base_models[i]

            if scaler is not None:
                X_test = scaler.transform(X_test)

            if hasattr(model, 'predict_proba'):
                test_predictions[:, i] = model.predict_proba(X_test)[:, 1]
            else:
                test_predictions[:, i] = model.predict(X_test).astype(float)

        # Meta-learner prediction
        proba = self.meta_learner.predict_proba(test_predictions)[:, 1]
        return proba

    def predict(self, test_smiles=None, threshold=0.5):
        """Generate class predictions on test data."""
        proba = self.predict_proba(test_smiles)
        return (proba >= threshold).astype(int)
