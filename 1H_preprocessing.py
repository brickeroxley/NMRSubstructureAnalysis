import os
import numpy as np
import pandas as pd
import glob
import h5py
from scipy.interpolate import interp1d

# 1H NMRPreprocessor class for normalizing spectra output
class H1NMRPreprocessor:
    def __init__(self, min_shift=-2.0, max_shift=12.0, num_points=28000):
        # Range from -2 to 12 ppm with 28000 points for increased resolution
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.num_points = num_points
        self.shift_step = (max_shift - min_shift) / num_points
    
    # Processing a singular CSV file that has been exported from MestReNova
    def process_file(self, file_path):
        try:
            # Extract molecule name from first line
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                molecule_name = first_line.split(',')[0]  
            
            df = pd.read_csv(file_path, names=[0, 1, 2, 3, 4, 5, 6, 7], header=None)
            separator_indices = df.index[df.iloc[:, 0].astype(str).str.contains('######')].tolist()
            
            # Extract spectral data discarding imaginary intensity values
            spectral_data = []
            for i in range(separator_indices[0] + 1, separator_indices[1]):
                row = df.iloc[i]
                shift = float(row[0])
                real = float(row[1])
                spectral_data.append([shift, real])
            
            # Convert to numpy array
            spectral_data = np.array(spectral_data)
            
            # Process the spectrum into a standardized format
            processed_spectrum = self.standardize_spectrum(spectral_data)
            
            # Extract peak data from the bottom of the CSV file 
            peak_data = []
            peak_table_start = separator_indices[1] + 2
            
            if peak_table_start < len(df):
                for i in range(peak_table_start, len(df)):
                    row = df.iloc[i]
                    if len(row) >= 7 and pd.notna(row[0]) and pd.notna(row[1]):
                        try:
                            # Peak processing including peak splitting patters, ppm ranges, center, and delta values
                            idx = int(row[0])
                            category = row[1]
                            range_max = float(row[2])
                            range_min = float(row[3])
                            n_hydrogens = float(row[4])
                            centroid = float(row[5])
                            delta = float(row[6])
                            
                            # Encode multiplicity as one-hot for ML pipeline (categorical to numerical)
                            one_hot = [0, 0, 0, 0]
                            if category == 's': one_hot = [1, 0, 0, 0]
                            elif category == 'd': one_hot = [0, 1, 0, 0]
                            elif category == 't': one_hot = [0, 0, 1, 0]
                            elif category == 'q': one_hot = [0, 0, 0, 1]
                            
                            peak_data.append({
                                'multiplicity': category,
                                'range_max': range_max,
                                'range_min': range_min,
                                'n_hydrogens': n_hydrogens,
                                'centroid': centroid,
                                'delta': delta,
                                'multiplicity_onehot': one_hot
                            })
                        except (ValueError, TypeError):
                            continue
            
            return {
                'molecule_name': molecule_name,
                'spectrum': processed_spectrum,
                'peak_data': peak_data
            }
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None
    
    # Transforms the raw spectral data into a standardized format with a fixed resolution and range
    def standardize_spectrum(self, spectral_data):
        # Extract shift and real components (column 1 and 2)
        shifts = spectral_data[:, 0]
        reals = spectral_data[:, 1]
        
        # Generate standardized shift values over the specified resolution
        std_shifts = np.linspace(self.min_shift, self.max_shift, self.num_points)
        
        # Sort data by shift values ensuring proper interpolation
        sort_idx = np.argsort(shifts)
        shifts_sorted = shifts[sort_idx]
        reals_sorted = reals[sort_idx]
        
        # Create interpolation function
        real_interp = interp1d(shifts_sorted, reals_sorted, kind='linear', 
                              bounds_error=False, fill_value=0)
        
        # Generate interpolated data between the generated values over the specified resolution and the intesity values from the spectral output
        real_spectrum = real_interp(std_shifts)
        
        # Normalize the spectrum so that the maximum intensity value is 1
        real_max = np.max(np.abs(real_spectrum)) if np.max(np.abs(real_spectrum)) > 0 else 1.0
        real_spectrum = real_spectrum / real_max
        
        return real_spectrum
    
    # Processing all CSV files that have been generated from the molecule dataset
    def process_dataset(self, input_dir, output_path, batch_size=1000):
        # Get all CSV files
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        total_files = len(csv_files)
        print(f"Found {total_files} CSV files to process")
        
        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Create datasets for spectra
            spectra_dataset = f.create_dataset(
                'spectra', 
                shape=(total_files, self.num_points),
                dtype=np.float32,
                chunks=(1, self.num_points),
                compression='gzip'
            )
            
            # Storing molecular names
            dt = h5py.special_dtype(vlen=str)
            molecule_names = f.create_dataset('molecule_names', (total_files,), dtype=dt)
            
            # Storing peak information obtained from the end of the CSV files
            peak_data_dt = h5py.special_dtype(vlen=np.dtype('float32'))
            peak_data = f.create_dataset('peak_data', (total_files,), dtype=peak_data_dt)
            
            # Process files in batches
            num_processed = 0
            num_errors = 0
            num_batches = (total_files + batch_size - 1) // batch_size
            
            print(f"Processing files in {num_batches} batches of size {batch_size}")
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_files)
                batch_files = csv_files[batch_start:batch_end]
                
                print(f"Processing batch {batch_idx+1}/{num_batches} (files {batch_start+1}-{batch_end} of {total_files})...")
                
                batch_processed = 0
                batch_errors = 0
                
                for i, file_path in enumerate(batch_files):
                    idx = batch_start + i
                    try:
                        result = self.process_file(file_path)
                        
                        if result is not None:
                            # Store spectrum
                            spectra_dataset[idx] = result['spectrum']
                            molecule_names[idx] = result['molecule_name']
                            
                            # Store peak data
                            if result['peak_data']:
                                peak_array = np.array([
                                    [p['centroid'], p['n_hydrogens']] + p['multiplicity_onehot']
                                    for p in result['peak_data']
                                ], dtype=np.float32).flatten()
                                peak_data[idx] = peak_array
                            else:
                                peak_data[idx] = np.array([], dtype=np.float32)
                            
                            batch_processed += 1
                        else:
                            batch_errors += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        batch_errors += 1
                
                num_processed += batch_processed
                num_errors += batch_errors
                
                # Print batch summary
                print(f"Batch {batch_idx+1} complete: {batch_processed} files processed, {batch_errors} errors")
                print(f"Overall progress: {num_processed}/{total_files} files processed ({num_processed/total_files*100:.1f}%)")
            
            f.attrs['min_shift'] = self.min_shift
            f.attrs['max_shift'] = self.max_shift
            f.attrs['num_points'] = self.num_points
            f.attrs['num_processed'] = num_processed
            f.attrs['num_errors'] = num_errors
            
            print(f"\nProcessing complete: {num_processed} files processed, {num_errors} errors")


if __name__ == "__main__":
    preprocessor = H1NMRPreprocessor()
    
    test_file = "Simulated_Spectra/simulation_out/1H/1H_1.csv"  
    result = preprocessor.process_file(test_file)

    if result:
        print(f"\nSuccessfully processed {test_file}")
        print(f"Molecule: {result['molecule_name']}")
        print(f"Spectrum shape: {result['spectrum'].shape}")
        print(f"Number of peaks: {len(result['peak_data'])}")
        
        for i, peak in enumerate(result['peak_data']):
            print(f"Peak {i+1}: {peak['multiplicity']} at {peak['centroid']} ppm, {peak['n_hydrogens']} H")
    
    # Process the full dataset
    input_dir = "E:/Simulated_Spectra/simulation_out/1H"
    output_path = "Simulated_Spectra/simulation_out/1H_processed/preprocessed_h_nmr_data.h5"
    preprocessor.process_dataset(input_dir, output_path)