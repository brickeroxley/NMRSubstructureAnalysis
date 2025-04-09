import pandas as pd
import h5py
from rdkit import Chem

# Generates molecular formula for each molecule by counting the elements in each SMILES string
class MolecularFormulaGenerator:
    def __init__(self):
        # Only four elements are contained in all the molecules
        self.elements = ['C', 'H', 'N', 'O']
    
    # Reading SMILES txt file and separating each molecule into a list
    def read_smiles_file(self, filename):
        with open(filename, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        print(f"Successfully read {len(smiles_list)} SMILES strings from {filename}")
        return smiles_list

    # Extracts element counts for each molecule in the SMILES list and stores them in a dictionary 
    def extract_element_counts(self, smiles):
        # Using Chem library to obtain molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to convert SMILES to molecule: {smiles}")
            return None
        
        # Add hydrogens explicitly for accurate H count
        mol = Chem.AddHs(mol)
        
        # Initialize results dictionary with molecule name
        results = {'SMILES': smiles}
        
        # Count elements
        for element in self.elements:
            atom_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
            count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atom_num)
            results[element] = count
        
        return results
    
    # Driver function which processes the list of extracted SMILE strings for element counts
    def process_smiles_list(self, smiles_list):
        results = []
        
        # Process each SMILES string
        counter = 0
        total = len(smiles_list)
        
        for smiles in smiles_list:
            counter += 1
            if counter % 1000 == 0:
                print(f"Processing molecule {counter}/{total}")
                
            counts = self.extract_element_counts(smiles)
            if counts:
                results.append(counts)
        
        print(f"Successfully processed {len(results)} of {len(smiles_list)} molecules")
        
        # Convert the list of dictionaries to dataframe
        df = pd.DataFrame(results)
        
        # Set SMILES as index
        df.set_index('SMILES', inplace=True)
        
        return df
    
    # Saving the dataframe to a CSV file in order to inspect the information processed
    def save_to_csv(self, df, output_file):
        try:
            df.to_csv(output_file)
            print(f"Saved element counts to CSV: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving to CSV {output_file}: {e}")
            return False
    
    # Saving the dataframe to an HDF5 file which can be used within the ML architecture
    def save_to_hdf5(self, df, output_file):
        try:
            with h5py.File(output_file, 'w') as h5f:
                # Create a dataset for each element column
                for element in self.elements:
                    if element in df.columns:
                        h5f.create_dataset(element, data=df[element].values)
                
                # Store SMILES strings as a dataset
                dt = h5py.string_dtype()
                smiles_list = list(df.index)
                smiles_ds = h5f.create_dataset('smiles', shape=(len(smiles_list),), dtype=dt)
                for i, smi in enumerate(smiles_list):
                    smiles_ds[i] = smi
            
            print(f"Saved element counts to HDF5: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving to HDF5 {output_file}: {e}")
            return False
    
    # Runs the entire pipeline including the txt file processing, SMILE list processing, and file saving
    def run(self, input_file, output_csv=None, output_h5=None):
        # Read SMILES from file
        smiles_list = self.read_smiles_file(input_file)
        if not smiles_list:
            return None
        
        # Process SMILES list to get element counts
        df = self.process_smiles_list(smiles_list)
        if df is None:
            return None
        
        # Save to CSV 
        if output_csv:
            self.save_to_csv(df, output_csv)
        
        # Save to HDF5
        if output_h5:
            self.save_to_hdf5(df, output_h5)
        
        return df

if __name__ == "__main__":
    generator = MolecularFormulaGenerator()
    
    input_file = "Selected_Structures/combined_structures.txt"  
    output_csv = "Molecular Formulas/csv/molecular_formulas.csv"   
    output_h5 = "Molecular Formulas/hdf5/molecular_formulas.h5"    
    
    df = generator.run(input_file, output_csv, output_h5)
    
    if df is not None:
        print("\nSummary Statistics:")
        print(f"Total molecules processed: {len(df)}")
        print("\nElement distribution:")
        for element in generator.elements:
            if element in df.columns:
                print(f"{element}: Mean = {df[element].mean():.2f}, Max = {df[element].max()}")