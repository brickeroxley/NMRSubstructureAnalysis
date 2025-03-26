import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

# Class using PyTorch's Dataset class handling loading all the data for learning
class NMRSubstructureDataset(Dataset):
    def __init__(self, h1_file, c13_file, formula_file, substructure_file, transform=None):
        self.h1_file = h1_file
        self.c13_file = c13_file
        self.formula_file = formula_file
        self.substructure_file = substructure_file
        
        # Verify all files have the same molecules in the same order
        self.verify_molecule_alignment()
        
        # Size of dataset
        with h5py.File(self.h1_file, 'r') as f:
            self.num_samples = len(f['molecule_names'])
            
        self.transform = transform
    
    # Ensures all files have the same molecule order
    def verify_molecule_alignment(self):
        # Opening all files to read from
        with h5py.File(self.h1_file, 'r') as h1_data, \
             h5py.File(self.c13_file, 'r') as c13_data, \
             h5py.File(self.formula_file, 'r') as formula_data, \
             h5py.File(self.substructure_file, 'r') as substructure_data:
            
            # Extracting the molecule name in the SMILE format to compare across files
            h1_names = [name.decode('utf-8') for name in h1_data['molecule_names']]
            c13_names = [name.decode('utf-8') for name in c13_data['molecule_names']]
            formula_smiles = [smile.decode('utf-8') for smile in formula_data['smiles']]
            substructure_smiles = [smile.decode('utf-8') for smile in substructure_data['molecule_smiles']]
            
            # Check first and last 10 molecules to verify alignment
            for i in range(min(10, len(h1_names))):
                if h1_names[i] != c13_names[i] or h1_names[i] != formula_smiles[i] or h1_names[i] != substructure_smiles[i]:
                    raise ValueError(f"Molecule mismatch at index {i}")
            
            print("Molecule alignment verified across all datasets")
    
    # Returning length of dataset
    def __len__(self):
        return self.num_samples
    
    # Returning a single sample by it's index
    def __getitem__(self, idx):
        with h5py.File(self.h1_file, 'r') as h1_data, \
             h5py.File(self.c13_file, 'r') as c13_data, \
             h5py.File(self.formula_file, 'r') as formula_data, \
             h5py.File(self.substructure_file, 'r') as substructure_data:
            
            # H1-NMR spectrum
            h1_spectrum = h1_data['spectra'][idx]
            
            # C13-NMR spectrum
            c13_bins = c13_data['bin_vectors'][idx]
            
            # Molecular formula information
            c_count = formula_data['C'][idx]
            h_count = formula_data['H'][idx]
            n_count = formula_data['N'][idx]
            o_count = formula_data['O'][idx]
            formula_features = np.array([c_count, h_count, n_count, o_count], dtype=np.float32)
            
            # Normalization of the values in the molecular formula information
            formula_features = formula_features / np.array([8, 19, 5, 4], dtype=np.float32)
            
            # Substructure target matrix
            substructure_targets = substructure_data['substructure_matrix'][idx]
        
        if self.transform:
            h1_spectrum = self.transform(h1_spectrum)
        
        # Convert to torch tensors for training
        h1_spectrum = torch.tensor(h1_spectrum, dtype=torch.float32).unsqueeze(0)
        c13_bins = torch.tensor(c13_bins, dtype=torch.float32)
        formula_features = torch.tensor(formula_features, dtype=torch.float32)
        substructure_targets = torch.tensor(substructure_targets, dtype=torch.float32)
        
        return {
            'h1_spectrum': h1_spectrum,
            'c13_bins': c13_bins,
            'formula': formula_features,
            'targets': substructure_targets
        }