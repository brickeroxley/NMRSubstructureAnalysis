import torch
import torch.nn as nn

# Torch neural network class which inherits PyTorch's nn.Module class  
class NMRSubstructurePredictor(nn.Module):
    def __init__(self, num_substructures=422):
        super(NMRSubstructurePredictor, self).__init__()
        
        # Branch 1: H1-NMR Spectral Processing (Narrow patterns)
        # Increasing channel dimensions and small kernel sizes for detailed feature extraction
        # ReLU for non-linearity and MaxPooling to reduce dimensionality
        self.h1_conv_fine = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=3, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Branch 1: H1-NMR Spectral Processing (Broad patterns)
        # Similar to narrow set-up but with larger kernal sizes for a greater breadth in feature extraction
        # ReLU for non-linearity and MaxPooling to reduce dimensionality
        self.h1_conv_broad = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=8, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Branch 2: C13-NMR and Molecular Formula Processing
        # 40 C13 binned binary points + 4 formula counts (C, H, N, O)
        # Processing of lower-dimensional data to aid the 1H NMR data
        self.c13_formula_fc = nn.Sequential(
            nn.Linear(40 + 4, 64),  
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Calculate the actual output sizes of the CNN branches dynamically
        # Runs a dummy tensor through the NN and records output size of each branch
        # Uses this to set output sizes for the actual model run 
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28000)
            h1_fine_out = self.h1_conv_fine(dummy_input)
            h1_broad_out = self.h1_conv_broad(dummy_input)
            
            self.h1_fine_output_size = h1_fine_out.view(1, -1).size(1)
            self.h1_broad_output_size = h1_broad_out.view(1, -1).size(1)
            print(f"H1 fine branch output size: {self.h1_fine_output_size}")
            print(f"H1 broad branch output size: {self.h1_broad_output_size}")
            
        # Combines all branches of the network to predict substructures
        # Dropout layers help to prevent overfitting
        # Fully connected layers used to reduce dimensionality until final prediction
        combined_input_size = self.h1_fine_output_size + self.h1_broad_output_size + 128
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_substructures)
        )
    
    # Defines how data flows through the network
    def forward(self, h1_spectrum, c13_bins, formula):
        # Process H1-NMR through both branches
        h1_fine_features = self.h1_conv_fine(h1_spectrum)
        h1_broad_features = self.h1_conv_broad(h1_spectrum)
        
        # Flatten the branch outputs
        h1_fine_features = h1_fine_features.view(h1_fine_features.size(0), -1)
        h1_broad_features = h1_broad_features.view(h1_broad_features.size(0), -1)
        
        # Process C13-NMR bins and molecular formula
        c13_formula_combined = torch.cat([c13_bins, formula], dim=1)
        c13_formula_features = self.c13_formula_fc(c13_formula_combined)
        
        combined_features = torch.cat([h1_fine_features, h1_broad_features, c13_formula_features], dim=1)
        output = self.combined_fc(combined_features)
        
        # Apply sigmoid activation for multi-label classification representing probability of substructure being present 
        return torch.sigmoid(output)
    
    # Initializing weights using Kaiming normal initialization method for ReLU activation
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)