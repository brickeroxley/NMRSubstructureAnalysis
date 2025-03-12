import os
import numpy as np
import pandas as pd
import glob
import h5py

# 13C NMRPreprocessor class for normalizing spectra output
class C13NMRPreprocessor:
    def __init__(self, min_shift=-20.0, max_shift=230.0, num_bins=40):
        # Range from -20 to 230 to match synthetic spectra output
        self.min_shift = min_shift
        self.max_shift = max_shift
        
        # 40 binary bins detailing whether a peak is present in that region or not
        # Spectral intensity is much less important for 13C NMR with this purpose
        self.num_bins = num_bins
        self.bin_size = (max_shift - min_shift) / num_bins
        self.bin_edges = np.linspace(min_shift, max_shift, num_bins + 1)
    
    # Processing a singular CSV file that has been exported from MestReNova
    def process_file(self, file_path):
        try:
            # Extract molecule name from first line
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                molecule_name = first_line.split(',')[0]
            
            df = pd.read_csv(file_path, names=[0, 1, 2], header=None)
            separator_indices = df.index[df.iloc[:, 0].astype(str).str.contains('######')].tolist()
            
            # Extract peaks from the peak table at the end of the CSV file
            peak_data = []
            if len(separator_indices) >= 2:
                peak_table_start = separator_indices[1] + 2                
                if peak_table_start < len(df):
                     for i in range(peak_table_start, len(df)):
                        row = df.iloc[i]
                        if len(row) >= 3 and pd.notna(row[0]) and pd.notna(row[1]):
                            try:
                                # Extra ppm peak shift from the peak table
                                delta = float(row[2])
                                peak_data.append(delta)
                            except (ValueError, TypeError):
                                continue
            
            # Create binary bin vector
            bin_vector = self.bin_peaks(peak_data)
            
            return {
                'molecule_name': molecule_name,
                'bin_vector': bin_vector,
                'peak_data': peak_data
            }
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # Creating a binary binned vector for the peak values obtained from the spectrum
    def bin_peaks(self, peak_shifts):
        # Initialize bin vector with zeros
        bin_vector = np.zeros(self.num_bins, dtype=np.int8)
        
        # Assign peaks to bins
        for shift in peak_shifts:
            if shift < self.min_shift or shift >= self.max_shift:
                continue
                
            # Calculate bin index
            bin_idx = int((shift - self.min_shift) // self.bin_size)
            
            # Stop index from going outside the bin vector
            if bin_idx == self.num_bins:
                bin_idx = self.num_bins - 1
                
            # Set bin index value to 1 (peak present)
            bin_vector[bin_idx] = 1
        
        return bin_vector
    
    # Processing entire batch of CSV files into a HDF5 dataset with binned peak information 
    def process_dataset(self, input_dir, output_path, batch_size=1000):
        # Get all CSV files
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        total_files = len(csv_files)
        print(f"Found {total_files} CSV files to process")
        
        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Create dataset for binary bin vectors
            bin_vectors_dataset = f.create_dataset(
                'bin_vectors',
                shape=(total_files, self.num_bins),
                dtype=np.int8,
                chunks=(100, self.num_bins),
                compression='gzip'
            )
            
            # Create dataset for molecule names
            dt = h5py.special_dtype(vlen=str)
            molecule_names = f.create_dataset('molecule_names', (total_files,), dtype=dt)
            
            # Create dataset for peak data
            peak_data_dt = h5py.special_dtype(vlen=np.dtype('float32'))
            peak_data = f.create_dataset('peak_data', (total_files,), dtype=peak_data_dt)
            
            # Process CVS files in batches
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
                            # Store binary bin vector
                            bin_vectors_dataset[idx] = result['bin_vector']
                            molecule_names[idx] = result['molecule_name']
                            
                            # Store peak data
                            if result['peak_data']:
                                peak_data[idx] = np.array(result['peak_data'], dtype=np.float32)
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
            f.attrs['num_bins'] = self.num_bins
            f.attrs['bin_size'] = self.bin_size
            f.attrs['bin_edges'] = self.bin_edges
            f.attrs['num_processed'] = num_processed
            f.attrs['num_errors'] = num_errors
            
            print(f"\nProcessing complete: {num_processed} files processed, {num_errors} errors")


if __name__ == "__main__":
    preprocessor = C13NMRPreprocessor()
    
    test_file = "Simulated_Spectra/simulation_out/13C/13C_1.csv"
    result = preprocessor.process_file(test_file)

    if result:
        print(f"\nSuccessfully processed {test_file}")
        print(f"Molecule: {result['molecule_name']}")
        print(f"Spectrum shape: {result['bin_vector'].shape}")
        print(f"Length of Binned spectrum: {len(result['bin_vector'])}")
        print(f"Binned spectrum: {result['bin_vector']}")
        print(f"Peak data: {result['peak_data']}")

    # Process the full dataset
    input_dir = "E:/Simulated_Spectra/simulation_out/13C"
    output_path = "Simulated_Spectra/simulation_out/13C_processed/preprocessed_c_nmr_data.h5"
    preprocessor.process_dataset(input_dir, output_path)