#!/usr/bin/env python
"""
Train Final Model and Evaluate on Test Set
===========================================

Single-step script to train final model and evaluate on predefined test set.
Use AFTER cross-validation has identified the best model/descriptor combination.

Takes both training and test data in one go:
1. Trains model on ALL training data
2. Evaluates on test set
3. Saves model and results

"""

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

from utils.descriptors import DescriptorGenerator
from utils.models import ModelFactory, calculate_metrics
from utils.ensemble import StackingEnsemble, select_top_k

warnings.filterwarnings('ignore')


# ============================================================================
# Helper Functions
# ============================================================================

def find_best_model(results_dir, metric='ROC_AUC'):
    """Find best model/descriptor combination from CV results."""
    results_file = Path(results_dir) / 'results_summary.csv'
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            "Run cross-validation first with pipeline.py"
        )
    
    df = pd.read_csv(results_file)
    
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not in results. Available: {df.columns.tolist()}"
        )
    
    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]
    
    descriptor = best_row['Descriptor']
    model = best_row['Model']
    
    print(f"\nBEST MODEL SELECTION (by {metric})")
    print(f"  Descriptor: {descriptor}")
    print(f"  Model: {model}")
    print(f"  CV {metric}: {best_row[metric]:.4f}")
    
    return descriptor, model


def load_and_validate_data(filepath, dataset_name="Data"):
    """Load and validate SMILES data with labels."""
    print(f"\nLOADING {dataset_name.upper()}")
    
    df = pd.read_csv(filepath)
    print(f"  Total samples: {len(df)}")
    
    # Validate columns
    if 'SMILES' not in df.columns or 'CLASS' not in df.columns:
        raise ValueError(
            f"CSV must have 'SMILES' and 'CLASS' columns. "
            f"Found: {df.columns.tolist()}"
        )
    
    # Validate SMILES
    valid_smiles = []
    valid_labels = []
    
    for smi, label in tqdm(zip(df['SMILES'], df['CLASS']), total=len(df), desc="  Validating SMILES", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_labels.append(label)

    print(f"  Valid SMILES: {len(valid_smiles)}")
    print(f"  Invalid SMILES: {len(df) - len(valid_smiles)}")
    
    if len(valid_smiles) == 0:
        raise ValueError(f"No valid SMILES found in {filepath}")
    
    # Check class balance
    labels = np.array(valid_labels)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    print(f"  Class 0: {n_neg} ({100*n_neg/len(labels):.1f}%)")
    print(f"  Class 1: {n_pos} ({100*n_pos/len(labels):.1f}%)")
    
    return valid_smiles, labels


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train final model and evaluate on test set (single step)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-select best model from CV results
    python train_test_best_model.py \\
        --train train.csv \\
        --test test.csv \\
        --results cv_results/ \\
        --output final_model/

    # Manual model/descriptor selection
    python train_test_best_model.py \\
        --train train.csv \\
        --test test.csv \\
        --descriptor Morgan \\
        --model XGBoost \\
        --output final_model/

    # Stacking ensemble of top 5 models
    python train_test_best_model.py \\
        --train train.csv \\
        --test test.csv \\
        --results cv_results/ \\
        --ensemble --top-k 5 \\
        --output final_model/

Input CSV format (both train and test):
    SMILES,CLASS
    CCO,0
    CC(=O)O,1
    ...

        """
    )
    
    # Required inputs
    parser.add_argument('--train', required=True,
                        help='Training data CSV (SMILES, CLASS)')
    parser.add_argument('--test', required=True,
                        help='Test data CSV (SMILES, CLASS)')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    
    # Model selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results',
                       help='CV results directory (auto-select best model)')
    group.add_argument('--descriptor',
                       help='Descriptor type (manual selection)')
    
    # Additional options
    parser.add_argument('--model',
                        help='Model type (required with --descriptor)')
    parser.add_argument('--metric', default='ROC_AUC',
                        help='Metric for best model selection (default: ROC_AUC)')
    parser.add_argument('--hyperparams',
                        help='Path to optimized_hyperparameters.json from Optuna')

    # Ensemble options
    parser.add_argument('--ensemble', action='store_true',
                        help='Use stacking ensemble of top-K models (requires --results)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top models for ensemble (default: 5)')

    args = parser.parse_args()

    # Validate
    if args.ensemble and not args.results:
        parser.error("--ensemble requires --results")
    if args.descriptor and not args.model:
        parser.error("--model required with --descriptor")
    if args.model and not args.descriptor:
        parser.error("--descriptor required with --model")
    
    start_time = datetime.now()
    
    if args.ensemble:
        print("\nSTACKING ENSEMBLE: TRAIN AND EVALUATE")
    else:
        print("\nTRAIN FINAL MODEL AND EVALUATE ON TEST SET")
    print(f"Started: {start_time:%Y-%m-%d %H:%M:%S}")

    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data (shared by both paths)
        train_smiles, y_train = load_and_validate_data(args.train, "Training Data")
        test_smiles, y_test = load_and_validate_data(args.test, "Test Data")

        if args.ensemble:
            _run_ensemble(args, output_dir, train_smiles, y_train,
                          test_smiles, y_test, start_time)
        else:
            _run_single_model(args, output_dir, train_smiles, y_train,
                              test_smiles, y_test, start_time)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def _run_ensemble(args, output_dir, train_smiles, y_train,
                  test_smiles, y_test, start_time):
    """Run stacking ensemble pipeline."""

    # Select top K
    top_k = select_top_k(args.results, k=args.top_k, metric=args.metric)

    print(f"\nSELECTED TOP {args.top_k} MODELS (by {args.metric})")
    for i, (desc, mdl, score) in enumerate(top_k, 1):
        print(f"  {i}. {desc} + {mdl} ({args.metric}={score:.4f})")

    base_configs = [(desc, mdl) for desc, mdl, _ in top_k]

    # Load hyperparameters
    hyperparams = {}
    if args.hyperparams:
        hp_path = Path(args.hyperparams)
        if hp_path.exists():
            with open(hp_path) as f:
                hyperparams = json.load(f)
            print(f"\nLoaded optimized hyperparameters from {hp_path}")
        else:
            print(f"\nHyperparams file not found: {hp_path}, using defaults")

    # Initialize descriptor generator
    cache_dir = output_dir / 'descriptor_cache'
    desc_gen = DescriptorGenerator(cache_dir=cache_dir)

    # Build ensemble
    print("\nBUILDING STACKING ENSEMBLE")
    ensemble = StackingEnsemble(
        base_configs=base_configs,
        desc_generator=desc_gen,
        hyperparams=hyperparams,
        n_folds=5,
        random_state=42,
    )

    oof_predictions = ensemble.fit(train_smiles, y_train, test_smiles)

    # OOF performance (meta-learner on training data)
    oof_proba = ensemble.meta_learner.predict_proba(oof_predictions)[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    oof_metrics = calculate_metrics(y_train, oof_pred, oof_proba)

    print("\nENSEMBLE OOF PERFORMANCE (training data)")
    for metric_name, value in oof_metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")

    # Evaluate on test set
    print("\nEVALUATING ENSEMBLE ON TEST SET")
    y_test_proba = ensemble.predict_proba()
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    print("\nENSEMBLE TEST SET PERFORMANCE")
    for metric_name, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")

    # Save
    print("\nSAVING RESULTS")

    # Save ensemble model
    ensemble_file = output_dir / 'stacking_ensemble.pkl'
    joblib.dump(ensemble, ensemble_file)
    print(f"  Ensemble: {ensemble_file.name}")

    # Save metadata
    metadata = {
        'type': 'stacking_ensemble',
        'base_models': [
            {'descriptor': d, 'model': m} for d, m in base_configs
        ],
        'meta_learner': 'LogisticRegression',
        'top_k': args.top_k,
        'selection_metric': args.metric,
        'hyperparameters': {
            f"{d}_{m}": hyperparams.get(f"{d}_{m}", 'defaults')
            for d, m in base_configs
        },
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(y_train),
        'test_samples': len(y_test),
    }
    metadata_file = output_dir / 'ensemble_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_file.name}")

    # Save test predictions
    pred_df = pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': y_test,
        'Predicted_Label': y_test_pred,
        'Predicted_Probability': y_test_proba,
        'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_test_pred]
    })
    pred_file = output_dir / 'ensemble_test_predictions.csv'
    pred_df.to_csv(pred_file, index=False)
    print(f"  Test predictions: {pred_file.name}")

    # Save test metrics
    metrics_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Value': list(test_metrics.values())
    })
    metrics_file = output_dir / 'ensemble_test_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Test metrics: {metrics_file.name}")

    # ----------------------------------------------------------------
    # Also train/evaluate the single best model for comparison
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BEST SINGLE MODEL (for comparison)")
    print("=" * 60)

    best_descriptor, best_model = find_best_model(args.results, args.metric)

    # Reuse descriptors from the ensemble's cache
    X_train_best = ensemble.descriptors_train.get(best_descriptor)
    X_test_best = ensemble.descriptors_test.get(best_descriptor)

    if X_train_best is None:
        # Descriptor wasn't in the ensemble's top-K; generate it
        X_train_best, X_test_best = ensemble._generate_descriptors(
            best_descriptor, train_smiles, test_smiles
        )

    # Scale if needed
    if best_descriptor.lower() == 'mordred':
        best_scaler = MinMaxScaler()
        X_train_best_scaled = best_scaler.fit_transform(X_train_best)
        X_test_best_scaled = best_scaler.transform(X_test_best)
    else:
        best_scaler = None
        X_train_best_scaled = X_train_best
        X_test_best_scaled = X_test_best

    # Load hyperparameters for the best model
    best_params = hyperparams.get(f"{best_descriptor}_{best_model}", {})
    if best_params:
        print(f"  Using optimized hyperparameters: {best_params}")

    clf = ModelFactory.create(best_model, **best_params)
    clf.fit(X_train_best_scaled, y_train)

    y_best_pred = clf.predict(X_test_best_scaled)
    y_best_proba = clf.predict_proba(X_test_best_scaled)[:, 1]
    best_metrics = calculate_metrics(y_test, y_best_pred, y_best_proba)

    print(f"\nBEST SINGLE MODEL TEST PERFORMANCE ({best_descriptor} + {best_model})")
    for metric_name, value in best_metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")

    # Save best single model artifacts
    best_model_file = output_dir / f"{best_descriptor}_{best_model}_model.pkl"
    joblib.dump(clf, best_model_file)
    best_scaler_file = output_dir / f"{best_descriptor}_{best_model}_scaler.pkl"
    joblib.dump(best_scaler, best_scaler_file)

    best_pred_df = pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': y_test,
        'Predicted_Label': y_best_pred,
        'Predicted_Probability': y_best_proba,
        'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_best_pred]
    })
    best_pred_file = output_dir / 'best_model_test_predictions.csv'
    best_pred_df.to_csv(best_pred_file, index=False)

    best_metrics_df = pd.DataFrame({
        'Metric': list(best_metrics.keys()),
        'Value': list(best_metrics.values())
    })
    best_metrics_file = output_dir / 'best_model_test_metrics.csv'
    best_metrics_df.to_csv(best_metrics_file, index=False)

    # ----------------------------------------------------------------
    # Side-by-side comparison
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON: ENSEMBLE vs BEST SINGLE MODEL")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Ensemble':>12} {best_descriptor + ' + ' + best_model:>25} {'Diff':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*25} {'-'*10}")

    for metric_name in test_metrics:
        ens_val = test_metrics[metric_name]
        best_val = best_metrics[metric_name]
        if isinstance(ens_val, float) and isinstance(best_val, float):
            diff = ens_val - best_val
            sign = '+' if diff >= 0 else ''
            print(f"  {metric_name:<20} {ens_val:>12.4f} {best_val:>25.4f} {sign}{diff:>9.4f}")

    # Save comparison CSV
    comparison_rows = []
    for metric_name in test_metrics:
        ens_val = test_metrics[metric_name]
        best_val = best_metrics[metric_name]
        if isinstance(ens_val, float) and isinstance(best_val, float):
            comparison_rows.append({
                'Metric': metric_name,
                'Ensemble': ens_val,
                f'{best_descriptor}_{best_model}': best_val,
                'Difference': ens_val - best_val,
            })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_file = output_dir / 'ensemble_vs_best_model_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n  Comparison saved: {comparison_file.name}")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\nCOMPLETED SUCCESSFULLY!")
    print(f"Duration: {duration}")
    print(f"\nOutput directory: {args.output}")
    print(f"  Ensemble files:")
    print(f"    - {ensemble_file.name}")
    print(f"    - {metadata_file.name}")
    print(f"    - {pred_file.name}")
    print(f"    - {metrics_file.name}")
    print(f"  Best single model files:")
    print(f"    - {best_model_file.name}")
    print(f"    - {best_scaler_file.name}")
    print(f"    - {best_pred_file.name}")
    print(f"    - {best_metrics_file.name}")
    print(f"  Comparison:")
    print(f"    - {comparison_file.name}")


def _run_single_model(args, output_dir, train_smiles, y_train,
                      test_smiles, y_test, start_time):
    """Run single model pipeline (original behavior)."""

    # Determine model/descriptor
    if args.results:
        descriptor, model = find_best_model(args.results, args.metric)
    else:
        descriptor = args.descriptor
        model = args.model
        print("\nMODEL SELECTION")
        print(f"  Descriptor: {descriptor}")
        print(f"  Model: {model}")

    # Initialize descriptor generator
    cache_dir = output_dir / 'descriptor_cache'

    print("\nINITIALIZING DESCRIPTOR GENERATOR")
    desc_gen = DescriptorGenerator(cache_dir=cache_dir)
    print(f"  Descriptor type: {descriptor}")
    print(f"  Cache directory: {cache_dir}")

    # Generate descriptors
    print("\nGENERATING DESCRIPTORS")

    print(f"\nCombining train and test data for descriptor generation...")
    all_smiles = train_smiles + test_smiles
    print(f"  Total molecules: {len(all_smiles)}")

    print(f"Generating {descriptor} descriptors...")
    X_all = desc_gen.generate(descriptor, all_smiles)
    print(f"  Combined shape: {X_all.shape}")

    n_train = len(train_smiles)
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")

    # Train model
    print("\nTRAINING MODEL ON ALL TRAINING DATA")

    if descriptor.lower() == 'mordred':
        print(f"\nUsing MinMaxScaler for {descriptor}")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        print(f"\nNo scaling for {descriptor} (already normalized)")
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Load optimized hyperparameters if provided
    model_params = {}
    if args.hyperparams:
        hp_path = Path(args.hyperparams)
        if hp_path.exists():
            with open(hp_path) as f:
                all_params = json.load(f)
            key = f"{descriptor}_{model}"
            model_params = all_params.get(key, {})
            if model_params:
                print(f"\nUsing optimized hyperparameters: {model_params}")
            else:
                print(f"\nNo optimized params found for {key}, using defaults")
        else:
            print(f"\nHyperparams file not found: {hp_path}, using defaults")

    print(f"\nCreating {model} model...")
    clf = ModelFactory.create(model, **model_params)

    print(f"Training on {len(y_train)} samples...")
    clf.fit(X_train_scaled, y_train)
    print("  Training complete")

    # Training metrics
    y_train_pred = clf.predict(X_train_scaled)
    y_train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)

    print("\nTRAINING SET PERFORMANCE")
    for metric, value in train_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Evaluate on test set
    print("\nEVALUATING ON TEST SET")

    y_test_pred = clf.predict(X_test_scaled)
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    print("\nTEST SET PERFORMANCE")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Save
    print("\nSAVING RESULTS")

    model_file = output_dir / f"{descriptor}_{model}_model.pkl"
    joblib.dump(clf, model_file)
    print(f"  Model: {model_file.name}")

    scaler_file = output_dir / f"{descriptor}_{model}_scaler.pkl"
    joblib.dump(scaler, scaler_file)
    print(f"  Scaler: {scaler_file.name}")

    metadata = {
        'descriptor': descriptor,
        'model': model,
        'scaler_type': 'MinMaxScaler' if descriptor.lower() == 'mordred' else 'None',
        'hyperparameters': model_params if model_params else 'defaults',
        'timestamp': datetime.now().isoformat(),
        'model_file': model_file.name,
        'scaler_file': scaler_file.name,
        'training_samples': len(y_train),
        'test_samples': len(y_test),
        'feature_dim': X_train.shape[1]
    }

    metadata_file = output_dir / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_file.name}")

    pred_df = pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': y_test,
        'Predicted_Label': y_test_pred,
        'Predicted_Probability': y_test_proba,
        'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_test_pred]
    })
    pred_file = output_dir / 'test_predictions.csv'
    pred_df.to_csv(pred_file, index=False)
    print(f"  Test predictions: {pred_file.name}")

    metrics_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Value': list(test_metrics.values())
    })
    metrics_file = output_dir / 'test_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Test metrics: {metrics_file.name}")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\nCOMPLETED SUCCESSFULLY!")
    print(f"Duration: {duration}")
    print(f"\nOutput directory: {args.output}")
    print(f"  Files created:")
    print(f"    - {model_file.name}")
    print(f"    - {scaler_file.name}")
    print(f"    - {metadata_file.name}")
    print(f"    - {pred_file.name}")
    print(f"    - {metrics_file.name}")
    print(f"\nTest set performance:")
    print(f"  ROC-AUC: {test_metrics['ROC_AUC']:.4f}")
    print(f"  MCC: {test_metrics['MCC']:.4f}")
    print(f"  Accuracy: {test_metrics['Accuracy']:.4f}")


if __name__ == "__main__":
    main()
