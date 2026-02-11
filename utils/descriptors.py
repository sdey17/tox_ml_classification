"""
Descriptor Utilities
====================

Handles all molecular descriptor generation with caching support.

Supported descriptors:
- Morgan (ECFP4) fingerprints
- RDKit fingerprints
- MACCS keys
- Mordred descriptors
- ChemBERTa embeddings
- MolFormer embeddings
"""

import warnings
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

from mordred import Calculator, descriptors

import torch
from transformers import AutoTokenizer, AutoModel
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DescriptorGenerator:
    """
    Generate molecular descriptors from SMILES.
    """
    
    AVAILABLE_DESCRIPTORS = ['Morgan', 'RDKit', 'MACCS', 'Mordred', 'ChemBERTa', 'MolFormer']
    
    def __init__(self, morgan_radius=2, morgan_nbits=2048, cache_dir=None):
        """
        Initialize descriptor generator.
        
        Args:
            morgan_radius: Morgan fingerprint radius (default: 2 for ECFP4)
            morgan_nbits: Number of bits in fingerprints (default: 2048)
            cache_dir: Directory to cache descriptors (default: None, no caching)
        """
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        
        # Setup cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Descriptor cache enabled: {self.cache_dir}")
        
        # Initialize fingerprint generators
        self.rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=morgan_nbits)
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=morgan_radius,
            fpSize=morgan_nbits
        )
        
        # Transformer models
        self.chemberta_model = None
        self.chemberta_tokenizer = None
        self.molformer_model = None
        self.molformer_tokenizer = None
        
        self.device = DEVICE
    
    # ========================================================================
    # Caching Methods
    # ========================================================================
    
    def _create_cache_key(self, descriptor_type, smiles_list, **kwargs):
        """
        Create unique cache key from descriptor type and SMILES.

        Args:
            descriptor_type: Type of descriptor
            smiles_list: List of SMILES strings
            **kwargs: Additional parameters

        Returns:
            Cache filename
        """
        # Use original order for hashing (order matters for row alignment)
        smiles_str = '|'.join(smiles_list)
        
        # Create hash
        hash_obj = hashlib.md5(smiles_str.encode())
        smiles_hash = hash_obj.hexdigest()[:16]
        
        # Include parameters in filename
        params_str = '_'.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        if params_str:
            filename = f"{descriptor_type}_{params_str}_{smiles_hash}.pkl"
        else:
            filename = f"{descriptor_type}_{smiles_hash}.pkl"
        
        return filename
    
    def _save_to_cache(self, descriptor_type, smiles_list, descriptors, **kwargs):
        """
        Save descriptors to cache.
        
        Args:
            descriptor_type: Type of descriptor
            smiles_list: List of SMILES strings
            descriptors: Generated descriptors
            **kwargs: Additional parameters
        """
        if not self.cache_dir:
            return
        
        cache_key = self._create_cache_key(descriptor_type, smiles_list, **kwargs)
        cache_path = self.cache_dir / cache_key
        
        # Save with metadata
        cache_data = {
            'descriptor_type': descriptor_type,
            'descriptors': descriptors,
            'smiles_list': smiles_list,
            'n_molecules': len(smiles_list),
            'n_features': descriptors.shape[1] if descriptors.ndim > 1 else descriptors.shape[0],
            'parameters': kwargs
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"    Saved to cache: {cache_key}")
    
    def _load_from_cache(self, descriptor_type, smiles_list, **kwargs):
        """
        Load descriptors from cache if available.
        
        Args:
            descriptor_type: Type of descriptor
            smiles_list: List of SMILES strings
            **kwargs: Additional parameters
            
        Returns:
            Cached descriptors or None if not found
        """
        if not self.cache_dir:
            return None
        
        cache_key = self._create_cache_key(descriptor_type, smiles_list, **kwargs)
        cache_path = self.cache_dir / cache_key
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            print(f"    Loaded from cache: {cache_key}")
            return cache_data['descriptors']
        
        except Exception as e:
            print(f"    Warning: Failed to load cache: {e}")
            return None
    
    def clear_cache(self, descriptor_type=None):
        """
        Clear descriptor cache.
        
        Args:
            descriptor_type: If specified, only clear this type. Otherwise clear all.
        """
        if not self.cache_dir:
            print("No cache directory configured")
            return
        
        if descriptor_type:
            # Clear specific descriptor type
            pattern = f"{descriptor_type}_*.pkl"
            files = list(self.cache_dir.glob(pattern))
        else:
            # Clear all
            files = list(self.cache_dir.glob("*.pkl"))
        
        for file in files:
            file.unlink()
        
        print(f"Cleared {len(files)} cache file(s)")
    
    def get_cache_info(self):
        """
        Get information about cached descriptors.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir:
            return {'enabled': False}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        info = {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'descriptors': {}
        }
        
        # Count by descriptor type
        for file in cache_files:
            desc_type = file.stem.split('_')[0]
            info['descriptors'][desc_type] = info['descriptors'].get(desc_type, 0) + 1
        
        return info
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def smiles_to_mols(self, smiles_list):
        """
        Convert SMILES to RDKit molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (valid_mols, valid_indices)
        """
        mols = []
        valid_idx = []
        
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_idx.append(i)
        
        return mols, valid_idx
    
    # ========================================================================
    # Morgan Fingerprints (ECFP4)
    # ========================================================================
    
    def generate_morgan(self, smiles_list):
        """
        Generate Morgan (ECFP) fingerprints.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            numpy array of shape (n_molecules, n_bits)
        """
        mols, _ = self.smiles_to_mols(smiles_list)
        
        fps = []
        for mol in mols:
            arr = np.zeros(self.morgan_nbits, dtype=np.int8)
            fp = self.morgan_gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        
        return np.array(fps)
    
    # ========================================================================
    # RDKit Fingerprints
    # ========================================================================
    
    def generate_rdkit(self, smiles_list):
        """
        Generate RDKit fingerprints.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            numpy array of shape (n_molecules, n_bits)
        """
        mols, _ = self.smiles_to_mols(smiles_list)
        
        fps = []
        for mol in mols:
            arr = np.zeros(self.morgan_nbits, dtype=np.int8)
            fp = self.rdkit_gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        
        return np.array(fps)
    
    # ========================================================================
    # MACCS Keys
    # ========================================================================
    
    def generate_maccs(self, smiles_list):
        """
        Generate MACCS keys fingerprints.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            numpy array of shape (n_molecules, 166)
        """
        mols, _ = self.smiles_to_mols(smiles_list)
        
        fps = []
        for mol in mols:
            arr = np.zeros(166, dtype=np.int8)
            fp = MACCSkeys.GenMACCSKeys(mol)
            tmp = np.zeros(167, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, tmp)
            arr[:] = tmp[1:]  # Drop bit-0 (unused)
            fps.append(arr)
        
        return np.array(fps)
    
    # ========================================================================
    # Mordred Descriptors
    # ========================================================================
    
    def generate_mordred(self, smiles_list, clean=True):
        """
        Generate Mordred descriptors with optional cleaning.
        
        Args:
            smiles_list: List of SMILES strings
            clean: Whether to clean descriptors (remove NaN, low variance, high correlation)
            
        Returns:
            numpy array of shape (n_molecules, n_descriptors)
        """
        mols, _ = self.smiles_to_mols(smiles_list)
        
        # Calculate descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        desc_df = calc.pandas(pd.Series(mols))
        
        if clean:
            print(f"    Cleaning Mordred descriptors...")
            original_n = desc_df.shape[1]
            
            # Replace inf with NaN, Drop columns with any NaN, and Keep only numeric columns
            desc_df = desc_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1).select_dtypes(include=[np.number])
            
            # Remove low variance columns
            variances = desc_df.var()
            desc_df = desc_df[variances[variances > 0.01].index]
            
            # Remove highly correlated columns
            corr = desc_df.corr(method='spearman').abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if (upper[col] > 0.95).any()]
            desc_df = desc_df.drop(columns=to_drop)
            
            print(f"    Kept {desc_df.shape[1]} / {original_n} descriptors")
        
        return desc_df.values
    
    # ========================================================================
    # ChemBERTa Embeddings
    # ========================================================================
    
    def generate_chemberta(self, smiles_list, batch_size=32):
        """
        Generate ChemBERTa embeddings.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference
            
        Returns:
            numpy array of shape (n_molecules, 768)
        """
        # Load model
        if self.chemberta_model is None:
            print(f"    Loading ChemBERTa model on {self.device}...")
            self.chemberta_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.chemberta_model.to(self.device)
            self.chemberta_model.eval()
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                
                inputs = self.chemberta_tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(self.device)
                
                outputs = self.chemberta_model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_emb)
        
        return np.array(embeddings)
    
    # ========================================================================
    # MolFormer Embeddings
    # ========================================================================
    
    def generate_molformer(self, smiles_list, batch_size=32):
        """
        Generate MolFormer embeddings.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference
            
        Returns:
            numpy array of shape (n_molecules, 768)
        """
        # Load model
        if self.molformer_model is None:
            print(f"    Loading MolFormer model on {self.device}...")
            self.molformer_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
            self.molformer_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
            self.molformer_model.to(self.device)
            self.molformer_model.eval()
        
        # Generate embeddings
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                
                inputs = self.molformer_tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(self.device)
                
                outputs = self.molformer_model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_emb)
        
        return np.array(embeddings)
    
    # ========================================================================
    # Main Interface
    # ========================================================================
    
    def generate(self, descriptor_type, smiles_list, **kwargs):
        """
        Generate descriptors by type with caching support.
        
        Args:
            descriptor_type: One of ['Morgan', 'RDKit', 'MACCS', 'Mordred', 'ChemBERTa', 'MolFormer']
            smiles_list: List of SMILES strings
            **kwargs: Additional arguments for specific descriptor types
            
        Returns:
            numpy array of descriptors
        """
        # Check cache first
        cached = self._load_from_cache(descriptor_type, smiles_list, **kwargs)
        if cached is not None:
            return cached
        
        # Generate descriptors
        if descriptor_type == 'Morgan':
            descriptors = self.generate_morgan(smiles_list)
        
        elif descriptor_type == 'RDKit':
            descriptors = self.generate_rdkit(smiles_list)
        
        elif descriptor_type == 'MACCS':
            descriptors = self.generate_maccs(smiles_list)
        
        elif descriptor_type == 'Mordred':
            clean = kwargs.get('clean', True)
            descriptors = self.generate_mordred(smiles_list, clean=clean)
        
        elif descriptor_type == 'ChemBERTa':
            batch_size = kwargs.get('batch_size', 32)
            descriptors = self.generate_chemberta(smiles_list, batch_size=batch_size)
        
        elif descriptor_type == 'MolFormer':
            batch_size = kwargs.get('batch_size', 32)
            descriptors = self.generate_molformer(smiles_list, batch_size=batch_size)
        
        else:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")
        
        # Save to cache
        self._save_to_cache(descriptor_type, smiles_list, descriptors, **kwargs)
        
        return descriptors


# ============================================================================
# Convenience Functions
# ============================================================================

def get_available_descriptors():
    """Get list of available descriptor types."""
    return DescriptorGenerator.AVAILABLE_DESCRIPTORS


def generate_descriptors(descriptor_type, smiles_list, **kwargs):
    """
    Convenience function to generate descriptors.
    
    Args:
        descriptor_type: Descriptor type
        smiles_list: List of SMILES
        **kwargs: Additional arguments
        
    Returns:
        numpy array of descriptors
    """
    gen = DescriptorGenerator()
    return gen.generate(descriptor_type, smiles_list, **kwargs)
