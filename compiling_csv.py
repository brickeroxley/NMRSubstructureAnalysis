import os
import csv
import glob

def combine_structures_to_csv(input_pattern, output_file):
    all_smiles = []

    for file_path in glob.glob(input_pattern):
        with open(file_path, 'r') as file:
            all_smiles.extend(file.read().splitlines())

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Smiles'])
        for index, smiles in enumerate(all_smiles, start=1):
            writer.writerow([index, smiles])

    print(f"Combined {len(all_smiles)} structures into {output_file}")

def main():
    input_dir = "NMR Project/Selected_Structures"
    
    input_pattern = os.path.join(input_dir, "*.txt")
    
    output_file = "NMR Project/Selected_Structures/combined_structures.csv"

    combine_structures_to_csv(input_pattern, output_file)

if __name__ == "__main__":
    main()


