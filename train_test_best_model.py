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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              roc_auc_score, matthews_corrcoef, confusion_matrix)

from utils.descriptors import DescriptorGenerator
from utils.models import ModelFactory, calculate_metrics

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
    python train_and_test.py \\
        --train train.csv \\
        --test test.csv \\
        --results cv_results/ \\
        --output final_model/
    
    # Manual model/descriptor selection
    python train_and_test.py \\
        --train train.csv \\
        --test test.csv \\
        --descriptor Morgan \\
        --model XGBoost \\
        --output final_model/
    
    # Use different metric for best model selection
    python train_and_test.py \\
        --train train.csv \\
        --test test.csv \\
        --results cv_results/ \\
        --metric MCC \\
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
    
    args = parser.parse_args()
    
    # Validate
    if args.descriptor and not args.model:
        parser.error("--model required with --descriptor")
    if args.model and not args.descriptor:
        parser.error("--descriptor required with --model")
    
    start_time = datetime.now()
    
    print("\nTRAIN FINAL MODEL AND EVALUATE ON TEST SET")
    print(f"Started: {start_time:%Y-%m-%d %H:%M:%S}")
    
    try:
        # ====================================================================
        # 1. DETERMINE MODEL/DESCRIPTOR
        # ====================================================================
        if args.results:
            descriptor, model = find_best_model(args.results, args.metric)
        else:
            descriptor = args.descriptor
            model = args.model
            print("\nMODEL SELECTION")
            print(f"  Descriptor: {descriptor}")
            print(f"  Model: {model}")
        
        # ====================================================================
        # 2. LOAD DATA
        # ====================================================================
        train_smiles, y_train = load_and_validate_data(args.train, "Training Data")
        test_smiles, y_test = load_and_validate_data(args.test, "Test Data")
        
        # ====================================================================
        # 3. INITIALIZE DESCRIPTOR GENERATOR (once, with caching)
        # ====================================================================
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = output_dir / 'descriptor_cache'
        
        print("\nINITIALIZING DESCRIPTOR GENERATOR")
        desc_gen = DescriptorGenerator(cache_dir=cache_dir)
        print(f"  Descriptor type: {descriptor}")
        print(f"  Cache directory: {cache_dir}")
        

        # ====================================================================
        # 4. GENERATE DESCRIPTORS (TRAIN + TEST COMBINED)
        # ====================================================================
        print("\nGENERATING DESCRIPTORS")
        
        # Combine SMILES to ensure consistent feature space
        print(f"\nCombining train and test data for descriptor generation...")
        all_smiles = train_smiles + test_smiles
        print(f"  Total molecules: {len(all_smiles)}")
        
        # Generate descriptors on ALL data
        print(f"Generating {descriptor} descriptors...")
        X_all = desc_gen.generate(descriptor, all_smiles)
        print(f"  Combined shape: {X_all.shape}")
        
        # Split back into train/test
        n_train = len(train_smiles)
        X_train = X_all[:n_train]
        X_test = X_all[n_train:]
        print(f"  Train shape: {X_train.shape}")
        print(f"  Test shape: {X_test.shape}")


        # ====================================================================
        # 5. TRAIN MODEL
        # ====================================================================
        print("\nTRAINING MODEL ON ALL TRAINING DATA")

        # Scale (only Mordred, only on train data)
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
        
        # Create model using ModelFactory
        print(f"\nCreating {model} model...")
        clf = ModelFactory.create(model)
        
        # Train
        print(f"Training on {len(y_train)} samples...")
        clf.fit(X_train_scaled, y_train)
        print(" Training complete")
        
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
        
        # ====================================================================
        # 6. EVALUATE ON TEST SET
        # ====================================================================
        print("\nEVALUATING ON TEST SET")
        
        # X_test_scaled was already created in section 5
        y_test_pred = clf.predict(X_test_scaled)
        y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        print("\nTEST SET PERFORMANCE")
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # ====================================================================
        # 7. SAVE EVERYTHING
        # ====================================================================
        print("\nSAVING RESULTS")
        
        # Save model
        model_file = output_dir / f"{descriptor}_{model}_model.pkl"
        joblib.dump(clf, model_file)
        print(f" Model: {model_file.name}")
        
        # Save scaler
        scaler_file = output_dir / f"{descriptor}_{model}_scaler.pkl"
        joblib.dump(scaler, scaler_file)
        print(f" Scaler: {scaler_file.name}")
        
        # Save metadata
        metadata = {
            'descriptor': descriptor,
            'model': model,
            'scaler_type': 'MinMaxScaler' if descriptor.lower() == 'mordred' else 'None',
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
        print(f" Metadata: {metadata_file.name}")
        
        # Save test predictions
        pred_df = pd.DataFrame({
            'SMILES': test_smiles,
            'True_Label': y_test,
            'Predicted_Label': y_test_pred,
            'Predicted_Probability': y_test_proba,
            'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_test_pred]
        })
        pred_file = output_dir / 'test_predictions.csv'
        pred_df.to_csv(pred_file, index=False)
        print(f" Test predictions: {pred_file.name}")
        
        # Save test metrics
        metrics_df = pd.DataFrame({
            'Metric': list(test_metrics.keys()),
            'Value': list(test_metrics.values())
        })
        metrics_file = output_dir / 'test_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f" Test metrics: {metrics_file.name}")
        
        # ====================================================================
        # 8. SUMMARY
        # ====================================================================
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
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
