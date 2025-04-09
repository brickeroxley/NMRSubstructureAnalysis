from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np
from typing import List, Set
import random
import pandas as pd
import h5py

class SubstructureFinder:
    # Input is a txt file containing all chosen structures in SMILES format
    def __init__(self, smiles_file: str):
        self.smiles_list = self.read_smiles_file(smiles_file)
        self.molecules = [Chem.MolFromSmiles(s) for s in self.smiles_list]
        self.substructures = []
        self.molecule_labels = np.zeros((len(self.molecules), 0))
        
    # Reading the input file and separating each structure into a list format
    def read_smiles_file(self, filename: str) -> List[str]:
        with open(filename, 'r') as f:
            return [line.strip().split()[0] for line in f if line.strip()]
    
    # Writing found substructures to a separate txt file
    def write_substructures(self, output_file: str):
        with open(output_file, 'w') as f:
            for smarts in self.substructures:
                f.write(f"{smarts}\n")
    
    # Function to filter substructures that are not found in a certain percentage of the structure population    
    def filter_rare_substructures(self, min_percentage: float = 0.5):
        if len(self.substructures) == 0:
            return
            
        # Calculate frequency for each substructure
        frequencies = self.molecule_labels.sum(axis=0) / len(self.molecules) * 100
        
        # Find indices of substructures to keep
        keep_indices = np.where(frequencies >= min_percentage)[0]
        
        # Update matrix and substructures list
        self.molecule_labels = self.molecule_labels[:, keep_indices]
        self.substructures = [self.substructures[i] for i in keep_indices]
        
        print(f"\nFrequency filtering results:")
        print(f"Original number of substructures: {len(frequencies)}")
        print(f"Substructures remaining after {min_percentage}% threshold: {len(self.substructures)}")
        print(f"New matrix shape: {self.molecule_labels.shape}")
        
        return self.substructures
    
    # If a substructure has identical labeling as another substructure only the first one is kept
    def remove_duplicate_substructures(self):
        if len(self.substructures) <= 1:
            return
            
        labeling_patterns = {}
        
        for i, smarts in enumerate(self.substructures):
            pattern_tuple = tuple(self.molecule_labels[:, i])
            if pattern_tuple in labeling_patterns:
                labeling_patterns[pattern_tuple].append(i)
            else:
                labeling_patterns[pattern_tuple] = [i]
        
        # Keep only the first substructure for each unique labeling pattern
        indices_to_keep = [group[0] for group in labeling_patterns.values()]
        
        # Update matrix and substructures list
        self.molecule_labels = self.molecule_labels[:, indices_to_keep]
        self.substructures = [self.substructures[i] for i in indices_to_keep]
        
        print(f"Removed {len(self.substructures) - len(indices_to_keep)} duplicate substructures")
        print(f"Remaining substructures: {len(self.substructures)}")

    # Finding substructures that differentiate two random pairs of structures from one another
    def find_differentiating_substructures(self, num_comparisons: int = 1, min_frequency: float = 0.5) -> List[str]:
        print(f"Starting with {len(self.molecules)} molecules")
        
        # Iterating through the amount of comparisons delineated 
        for i in range(num_comparisons):
            print(f"\nComparison {i+1}/{num_comparisons}")
            
            mol1, mol2 = random.sample(self.molecules, 2)
            new_substructures = self.compare_molecules(mol1, mol2)
            print(f"Found {len(new_substructures)} potential new substructures")
            
            if new_substructures:
                self.update_substructures_and_labels(new_substructures)
        
        print(f"\nBefore frequency filtering:")
        print(f"Number of unique substructures: {len(self.substructures)}")
        print(f"Matrix shape: {self.molecule_labels.shape}")
        
        # Removing duplicate substructures with the same labeling patterns
        self.remove_duplicate_substructures()

        # Filter rare substructures
        if min_frequency > 0:
            self.filter_rare_substructures(min_frequency)
        
        return self.substructures
    
    # Comparing the two randomly selected molecules from the population
    def compare_molecules(self, mol1: Chem.Mol, mol2: Chem.Mol) -> Set[str]:
        # Generating the substructure fragments from each molecule
        fragments1 = self.generate_fragments(mol1)
        fragments2 = self.generate_fragments(mol2)
        
        # Find unique fragments
        unique_to_mol1 = fragments1 - fragments2
        unique_to_mol2 = fragments2 - fragments1
        
        return unique_to_mol1.union(unique_to_mol2)
    
    # Fragments generated from molecules using BRICS decomposition 
    def generate_fragments(self, mol: Chem.Mol) -> Set[str]:
        fragments = set()
        
        # BRICS decomposition implementation
        broken_bonds = list(BRICS.BRICSDecompose(mol))
        if broken_bonds:
            for frag in broken_bonds:
                try:
                    smarts = Chem.MolToSmarts(frag)
                    fragments.add(smarts)
                except:
                    continue

        # Generate atom-centered submolecules based on a specific radius from the centered atom
        for atom in mol.GetAtoms():
            for radius in [1, 2]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
                if env:
                    try:
                        submol = Chem.PathToSubmol(mol, env)
                        smarts = Chem.MolToSmarts(submol)
                        fragments.add(smarts)
                    except:
                        continue
        
        return fragments
    
    # Filtering fragments between 2 and 8 atoms to ensure they don't get too large
    def filter_fragments(self, smarts: str) -> bool:
        try:
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                return False
            
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 2 or num_atoms > 8:
                return False
            
            return True
        except:
            return False
    
    # Updating binary labels for each molecule in the population based on if it contains the given substructure
    def update_substructures_and_labels(self, new_substructures: Set[str]):
        # Filtering
        filtered_new = {s for s in new_substructures if self.filter_fragments(s)}
        
        # Check for new substructures that are not duplicates
        unique_new = []
        for smarts in filtered_new:
            if smarts not in self.substructures:
                unique_new.append(smarts)
        
        if not unique_new:
            return
        
        # Shaping the new matrix and labeling the new columns for the new substructures
        new_matrix = np.zeros((len(self.molecules), len(self.substructures) + len(unique_new)))
        
        if self.molecule_labels.shape[1] > 0:
            new_matrix[:, :self.molecule_labels.shape[1]] = self.molecule_labels
        
        # Iterating through each new substructure and molecule and determining if that substructure is in that molecule
        for i, smarts in enumerate(unique_new):
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                col_idx = len(self.substructures) + i
                for j, mol in enumerate(self.molecules):
                    if mol.HasSubstructMatch(pattern):
                        new_matrix[j, col_idx] = 1
        
        self.molecule_labels = new_matrix
        self.substructures.extend(unique_new)
        
        print(f"Added {len(unique_new)} new unique substructures")
        print(f"Current total: {len(self.substructures)}")
        print(f"Matrix shape: {self.molecule_labels.shape}")
    
    # Exporting all results to an excel file including binary substructure determination, substructure SMARTS strings, and population frequency
    def export_to_excel(self, output_file: str, max_smiles_length=20):
        shortened_smarts = [f"Sub_{i}" for i in range(len(self.substructures))]
        
        labels_df = pd.DataFrame(
            self.molecule_labels,
            columns=shortened_smarts,
            index=[f"{s[:max_smiles_length]}" for i, s in enumerate(self.smiles_list)]
        )
        
        mapping_df = pd.DataFrame({
            'Substructure_ID': shortened_smarts,
            'SMARTS': self.substructures
        })
        
        stats_df = pd.DataFrame({
            'Substructure_ID': shortened_smarts,
            'Frequency': labels_df.sum().values,
            'Percentage': (labels_df.sum().values / len(self.molecules) * 100).round(2)
        })
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            labels_df.to_excel(writer, sheet_name='Labels_Matrix')
            mapping_df.to_excel(writer, sheet_name='SMARTS_Mapping', index=False)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"Excel file saved to: {output_file}")
        return labels_df, mapping_df, stats_df
    
    # Function to export the substructure findings as a h5py dataset for ML integration
    def export_to_hdf5(self, output_file: str):
        try:
            with h5py.File(output_file, 'w') as f:
                # Store the binary labels matrix
                f.create_dataset('substructure_matrix', 
                                data=self.molecule_labels,
                                chunks=(min(100, self.molecule_labels.shape[0]), 
                                        min(100, self.molecule_labels.shape[1])),
                                compression='gzip')
                
                # Store the SMARTS patterns as attributes
                dt = h5py.special_dtype(vlen=str)
                smarts_ds = f.create_dataset('smarts_patterns', shape=(len(self.substructures),), dtype=dt)
                for i, smarts in enumerate(self.substructures):
                    smarts_ds[i] = smarts
                
                # Store molecule SMILES strings
                smiles_ds = f.create_dataset('molecule_smiles', shape=(len(self.smiles_list),), dtype=dt)
                for i, smiles in enumerate(self.smiles_list):
                    smiles_ds[i] = smiles
                
                # Store statistics as attributes
                frequencies = self.molecule_labels.sum(axis=0)
                percentages = (frequencies / len(self.molecules) * 100)
                
                f.create_dataset('substructure_frequencies', data=frequencies)
                f.create_dataset('substructure_percentages', data=percentages)
                
                # Add metadata
                f.attrs['num_molecules'] = len(self.molecules)
                f.attrs['num_substructures'] = len(self.substructures)
                
            print(f"HDF5 file saved to: {output_file}")
            return True
        
        except Exception as e:
            print(f"Error saving to HDF5: {e}")
            return False

if __name__ == "__main__":
    # Read molecules from txt file and find substructures
    finder = SubstructureFinder("Selected_Structures/combined_structures.txt")
    substructures = finder.find_differentiating_substructures(num_comparisons=10000, min_frequency=2)

    # Write substructures to a separate txt file
    finder.write_substructures("Substructures/combined_substructures.txt")

    print(f"Found {len(substructures)} unique substructures")

    hdf5_path = "Substructures/substructure_analysis.h5"
    finder.export_to_hdf5(hdf5_path)

    if len(finder.substructures) < 2000:  
        excel_path = "Substructures/substructure_analysis.xlsx"
        try:
            labels_df, mapping_df, stats_df = finder.export_to_excel(excel_path)
        except Exception as e:
            print(f"Excel export failed (likely due to size), but HDF5 file was saved: {e}")