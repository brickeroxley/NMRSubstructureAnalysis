import random
import os

def randomly_select_molecules(input_file, output_file, num_to_select):
    with open(input_file, 'r') as infile:
        molecules = infile.readlines()
    
    selected = random.sample(molecules, num_to_select)
    
    with open(output_file, 'w') as outfile:
        outfile.writelines(selected)

def main():
    base_dir = r"C:\Users\Brickhouse\VSCode\Python\NMR Project\Structures"
    output_dir = r"C:\Users\Brickhouse\VSCode\Python\NMR Project\Selected_Structures"

    to_select = 15194

    print(f"Selecting {to_select} molecules each from 9.cno and 10.cno")

    for file in ['9.cno.txt', '10.cno.txt']:
        input_file = os.path.join(base_dir, file)
        output_file = os.path.join(output_dir, f"Selected_{file}")
        randomly_select_molecules(input_file, output_file, to_select)
        print(f"Selected molecules from {file} saved to {output_file}")

    print("Process completed. Total molecules selected: 50,000")

if __name__ == "__main__":
    main()