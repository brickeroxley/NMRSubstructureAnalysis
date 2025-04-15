# Substructure Analysis of 1D NMR Spectra Using a Machine Learning Approach
This project aims to predict likely substructure constituents of unknown 
compounds based on routine 1D 1H and 13C NMR data alongside molecular formulae 
Unlike previous work, this project implements a dual stream ML architecture 
which analyzes both narrow and broad spectral nuances within the 1H NMR spectra. 
Which, when combined with the 13C NMR data and molecular formulae, produces a 
probability value related to whether or not that compound contains a given substructure.
Trained chemists will deconstruct the information within each input and deduce 
a small handful of possible substructures related to the parent molecule. 
This model attempts to analyze a multitude distinct substructures in a fraction of 
the time it would take a chemist in order to accelerate chemical discoveries and introduce 
autonomy into the pipeline. 

Data processing will be applied to a multitude of 1H and 13C NMR spectra that have 
been synthetically produced using MestReNova. A subset of the synthetically generated 
data will be used for validation and testing. In order to generate a list of substructures, 
a separate script will compare two molecules from the molecule set and extract similar 
constituents from each molecule. The methodology includes using BRICS decomposition and atom radius 
logic from Python’s RDKit library. This list will be narrowed down for a comprehensive 
compendium of substructures for attempted elucidation. The ML model will output 
probabilities for each substructure being present in the test molecules. 
In addition to this, statistical metrics indicating the model’s ability to predict 
the presence of certain key functional groups across the entire set will also be generated.

The architecture implemented for this task involves an input layer for the 1H NMR 
spectra which is then fed through a dual-branch architecture containing both fine and 
broad branches. The fine branch extracts key details about localized peak information 
and detailed spectral features. The board branch extracts more regional patterns and 
global spectral features. Another branch within the architecture takes the binned binary 
13C NMR data and the molecular formula for the molecule being tested. These are combined 
together through two fully connected layers and then merged with the output from both 
the 1H NMR fine and broad branches. All of the merged data is combined through three 
fully connected layers with dropout ratios to stave off overfitting. At the end of 
these layers a prediction is made for which of the substructures are present in the molecule. 

Running the Code:
The codebase comes with a sample set of 10 molecules, however any molecular set you want to use
would also work. Initial testing was done with molecules that are under 10 non-hydrogen atoms in 
length, but any length of molecule should work given the proper processing power and robust
training data. Paste your molecule list into the *combined_structures.txt* within the
*Selected_Structures* folder. Run the *substructures_from_structures.py* file to generate a list
of possible substructures from the molecular list. Once this has been completed, you need to run
the MestReNova simulation. This software can be downloaded from https://mestrelab.com/ on a free
trial and ran with the following instructions:

Before running the run_mestrenova_simulation.py file, ensure that the scripts contained in the 
“MestReNova scripts” folder are placed in the script folder within the directory containing your 
MestReNova executable. 

The following commands will detail how to use the run_mestrenova_simulation.py file:
1H NMR
run_mestrenova_simulation --smiles_csv Selected_Structures/combined_structures.csv --out_folder Selected_Structures/simulation_out/1H --sim_type 1H --mnova_path <Absolute path to your MestReNova executable> --script_path <Absolute path to the folder containing the MestReNova scripts>

13C NMR
run_mestrenova_simulation --smiles_csv Selected_Structures/combined_structures.csv --out_folder Selected_Structures/simulation_out/13C --sim_type 13C --mnova_path <Absolute path to your MestReNova executable> --script_path <Absolute path to the folder containing the MestReNova scripts>

Once the simulations are complete, run both the *1H_preprocessing.py*, *13C_preprocessing.py*,
and *molecular_formulas.py* files. This will generate the database files required for model training.
After all files are generated, run *main.py* to train the model and get statistical metrics and
various visualizations for substructure predictive performance. Hyperparameters are set at 100 epochs, 
16 batch size, seed of 8, learning rate of 0.001, decay of 1e-5, training size of 0.8, validation size
of 0.1, and training size of 0.1. All parameters can be changed depending on the needs of your dataset.
