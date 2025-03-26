import torch
import numpy as np
import random
import json
from sklearn.metrics import f1_score

# Setting seed for reproducibility of the results
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Saves the configuration containing the experimental parameters in a JSON file
def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

# Loads a previous configuration with experimental parameters from a JSON file
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

# Returns the count of trainable parameters in the model 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Determines optimal classification threshold (max F1 score) for each substructure
# Makes this decision dynamic rather than using a fixed threshold for all substructures
def optimize_thresholds(model, val_loader, device, num_substructures):
    model.eval()
    all_outputs = []
    all_targets = []
    
    # Collect all predictions and targets
    with torch.no_grad():
        for batch in val_loader:
            h1_spectrum = batch['h1_spectrum'].to(device)
            c13_bins = batch['c13_bins'].to(device)
            formula = batch['formula'].to(device)
            
            outputs = model(h1_spectrum, c13_bins, formula).cpu().numpy()
            targets = batch['targets'].cpu().numpy()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
    
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    # Find optimal threshold for each substructure
    optimal_thresholds = np.zeros(num_substructures)
    best_f1s = np.zeros(num_substructures)
    
    for i in range(num_substructures):
        best_f1 = 0
        best_threshold = 0.5
        
        # Try different thresholds from 0.1 to 0.9 with a 0.02 step size to find best F1 score
        for threshold in np.arange(0.1, 0.9, 0.02):
            pred = (all_outputs[:, i] > threshold).astype(float)
            f1 = f1_score(all_targets[:, i], pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[i] = best_threshold
        best_f1s[i] = best_f1
    
    print(f"Average best F1 score: {np.mean(best_f1s):.4f}")
    return optimal_thresholds