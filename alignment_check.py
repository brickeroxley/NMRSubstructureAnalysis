import h5py

# Load a few samples from each dataset
with h5py.File('Simulated_Spectra/simulation_out/1H_processed/preprocessed_h_nmr_data.h5', 'r') as f1h:
    h1_names = f1h['molecule_names'][:50000]  # First 10 molecule names

with h5py.File('Simulated_Spectra/simulation_out/13C_processed/preprocessed_c_nmr_data.h5', 'r') as f13c:
    c13_names = f13c['molecule_names'][:50000]

with h5py.File('Substructures/substructure_analysis.h5', 'r') as sub:
    sub_names = sub['molecule_smiles'][:50000]
    
with h5py.File('Molecular Formulas/hdf5/molecular_formulas.h5', 'r') as fmf:
    mf_names = fmf['smiles'][:50000]

# Compare names
print("1H names:", h1_names)
print("13C names:", c13_names)
print("Substructure names:", sub_names)
print("Formula names:", mf_names)

# Check if they match
match = all(h1_names[i] == c13_names[i] == sub_names[i] == mf_names[i] for i in range(50000))
print(f"Data alignment: {'OK' if match else 'MISALIGNED'}")