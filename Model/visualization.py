import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
import os
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw

# Plotting the progression of major matrics obtained during training
def plot_training_history(history, save_path='results'):
    """Plot training and validation metrics"""    
    os.makedirs(save_path, exist_ok=True)

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='darkgreen')
    plt.plot(history['val_loss'], label='Validation Loss', color='indigo')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}/loss_history.png')
    plt.close()
    
    # Plot average precision
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_ap'], label='Validation AP', color='darkgreen')
    plt.title('Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}/ap_history.png')
    plt.close()

    # Plot AUC
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_auc'], label='Validation AUC', color='darkgreen')
    plt.title('Area Under ROC Curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}/auc_history.png')
    plt.close()
    
    # Plot F1 score
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_f1'], label='Validation F1', color='darkgreen')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}/f1_history.png')
    plt.close()

# Visualize model predicitons on a random selection of molecules in the test set
def visualize_predictions(model, dataset, indices, device, substructure_file, save_dir='results/visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Load substructure SMARTS patterns
    with h5py.File(substructure_file, 'r') as f:
        smarts_patterns = [s.decode('utf-8') for s in f['smarts_patterns']]
    
    for idx in indices:
        sample = dataset[idx]
        
        h1_spectrum = sample['h1_spectrum'].unsqueeze(0).to(device)
        c13_bins = sample['c13_bins'].unsqueeze(0).to(device)
        formula = sample['formula'].unsqueeze(0).to(device)
        targets = sample['targets'].numpy()
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(h1_spectrum, c13_bins, formula).cpu().numpy()[0]
        
        # Get molecule name and SMILES
        with h5py.File(dataset.h1_file, 'r') as f:
            molecule_name = f['molecule_names'][idx].decode('utf-8')
        
        with h5py.File(dataset.substructure_file, 'r') as f:
            smiles = f['molecule_smiles'][idx].decode('utf-8')
        
        mol = Chem.MolFromSmiles(smiles)
        
        # Plot H1 spectrum
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        h1_data = sample['h1_spectrum'].numpy()[0]
        shift_values = np.linspace(-2, 12, 28000)
        plt.plot(shift_values, h1_data, color='darkgreen')
        plt.title(f'H1-NMR Spectrum - {molecule_name}')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.xlim([12, -2])
        
        # Plot molecule structure
        plt.subplot(2, 1, 2)
        mol_img = Draw.MolToImage(mol, size=(300, 300))
        plt.imshow(mol_img)
        plt.axis('off')
        plt.title(f'Structure: {smiles}')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/molecule_{idx}_spectrum.png')
        plt.close()

        plt.figure(figsize=(15, 10))
        # Find top predicted and actual substructures
        top_pred_indices = np.argsort(outputs)[-10:][::-1]
        top_target_indices = np.argsort(targets)[-10:][::-1]

        all_indices = list(set(top_pred_indices) | set(top_target_indices))
        all_indices = all_indices[:min(20, len(all_indices))]
        
        pred_values = [outputs[i] for i in all_indices]
        target_values = [targets[i] for i in all_indices]

        labels = [f"Substructure {i+1}" for i in range(len(all_indices))]

        # Plot as bar chart
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, target_values, width, label='Actual', color='darkgreen')
        plt.bar(x + width/2, pred_values, width, label='Predicted', color='indigo')
        
        plt.xlabel('Substructure')
        plt.ylabel('Probability')
        plt.title(f'Top Substructures - {molecule_name}')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/molecule_{idx}_substructures.png')
        plt.close()

        # Create substructure visualization table
        fig, ax = plt.figure(figsize=(15, len(all_indices) * 1.5)), plt.gca()
        
        ax.axis('off')
        col_width = [0.7, 0.15, 0.15]
        row_height = 1.0 / (len(all_indices) + 1)
        
        ax.text(0.35, 1 - row_height/2, "Substructure", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.35 + col_width[0], 1 - row_height/2, "Predicted", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.35 + col_width[0] + col_width[1], 1 - row_height/2, "Actual", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.axhline(y=1-row_height, color='black', linewidth=1)

        for i, substructure_idx in enumerate(all_indices):
            y_pos = 1 - row_height * (i + 1.5)
            
            # Draw substructure
            pattern = Chem.MolFromSmarts(smarts_patterns[substructure_idx])
            substructure_img = Draw.MolToImage(pattern, size=(300, 150))
            
            img_ax = fig.add_axes([0.05, y_pos - row_height*0.4, col_width[0] * 0.8, row_height * 0.8])
            img_ax.imshow(substructure_img)
            img_ax.axis('off')

            # Draw predicted value
            pred_val = outputs[substructure_idx]
            ax.text(0.35 + col_width[0], y_pos, f"{pred_val:.3f}", ha='center', va='center')
            
            # Draw actual value
            actual_val = targets[substructure_idx]
            ax.text(0.35 + col_width[0] + col_width[1], y_pos, f"{actual_val:.3f}", ha='center', va='center')

            if i < len(all_indices) - 1:
                ax.axhline(y=1 - row_height * (i + 2), color='gray', linewidth=0.5, linestyle=':')
        
        ax.axvline(x=col_width[0], color='black', linewidth=1, ymin=0, ymax=1)
        ax.axvline(x=col_width[0] + col_width[1], color='black', linewidth=1, ymin=0, ymax=1)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/molecule_{idx}_substructure_table.png')
        plt.close()

# Analyze the ratio of feature importance for all inputs to the model
def analyze_feature_importance(model, dataset, device, save_path='results/feature_importance.png'):    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    
    h1_spectrum = batch['h1_spectrum'].to(device)
    c13_bins = batch['c13_bins'].to(device)
    formula = batch['formula'].to(device)
    
    # Original predictions
    model.eval()
    with torch.no_grad():
        original_outputs = model(h1_spectrum, c13_bins, formula).cpu().numpy()
    
    # Predictions with zeroed H1-NMR
    h1_zeroed = torch.zeros_like(h1_spectrum).to(device)
    with torch.no_grad():
        h1_zeroed_outputs = model(h1_zeroed, c13_bins, formula).cpu().numpy()
    
    # Predictions with zeroed C13-NMR
    c13_zeroed = torch.zeros_like(c13_bins).to(device)
    with torch.no_grad():
        c13_zeroed_outputs = model(h1_spectrum, c13_zeroed, formula).cpu().numpy()
    
    # Predictions with zeroed formula
    formula_zeroed = torch.zeros_like(formula).to(device)
    with torch.no_grad():
        formula_zeroed_outputs = model(h1_spectrum, c13_bins, formula_zeroed).cpu().numpy()
    
    # Calculate impact
    h1_impact = np.mean(np.abs(original_outputs - h1_zeroed_outputs))
    c13_impact = np.mean(np.abs(original_outputs - c13_zeroed_outputs))
    formula_impact = np.mean(np.abs(original_outputs - formula_zeroed_outputs))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    impact_values = [h1_impact, c13_impact, formula_impact]
    features = ['H1-NMR', 'C13-NMR', 'Molecular Formula']
    
    plt.bar(features, impact_values, color=['darkgreen', 'indigo', 'darkblue'])
    plt.title('Feature Importance Analysis')
    plt.ylabel('Average Impact on Predictions')
    plt.ylim([0, max(impact_values) * 1.1])
    
    for i, v in enumerate(impact_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print("Feature Importance:")
    print(f"H1-NMR Impact: {h1_impact:.4f}")
    print(f"C13-NMR Impact: {c13_impact:.4f}")
    print(f"Molecular Formula Impact: {formula_impact:.4f}")

# Creates performance table for certain functional group substructures and their performance
def create_performance_table(test_results, substructure_file, save_path='results/performance_table.png'):
    # Load substructure SMARTS patterns and frequency data
    with h5py.File(substructure_file, 'r') as f:
        smarts_patterns = [s.decode('utf-8') for s in f['smarts_patterns']]
        frequencies = f['substructure_frequencies'][:]
    
    # Get metrics from test results
    class_ap = test_results['class_ap']
    class_f1 = test_results['class_f1']
    
    # Calculate accuracy for each substructure
    all_outputs = test_results['all_outputs']
    all_targets = test_results['all_targets']
    
    predictions = (all_outputs > 0.5).astype(np.float32)
    accuracy_scores = []
    correct_counts = []
    test_set_counts = []
    
    for i in range(all_targets.shape[1]):
        tp = ((predictions[:, i] == 1) & (all_targets[:, i] == 1)).sum()
        tn = ((predictions[:, i] == 0) & (all_targets[:, i] == 0)).sum()
        correct = tp + tn
        present = (all_targets[:, i] == 1).sum()
        total = len(all_targets[:, i])

        accuracy_scores.append(correct / total)
        correct_counts.append(tp)
        test_set_counts.append(present)
    
    # Obtain information for certain functional groups 
    all_indices = [10, 131, 19, 8, 13, 0, 23, 5, 59, 104, 266, 107, 6, 112]
    
    fig = plt.figure(figsize=(12, 15))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.suptitle('Performance of the substructure prediction model for selected substructures in test set',
                 fontsize=14, y=0.98)
    column_labels = ['Entry', 'Substructure', 'SMARTS string', 'Accuracy', 'F₁ score', 'PRC-AUC score', 
                    'Total in dataset', 'In test set', 'Correct pred.', 'Ratio']   
    row_labels = [f"{i+1}" for i in range(len(all_indices))] 

    n_rows, n_cols = len(all_indices) + 1, len(column_labels)

    # Plot table with metrics and substructures
    ax = plt.subplot(111)
    ax.axis('off')
    
    column_widths = [0.05, 0.15, 0.15, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.07]
    cum_widths = np.cumsum([0] + column_widths)
    
    for j, label in enumerate(column_labels):
        x = cum_widths[j] + column_widths[j]/2
        ax.text(x, 0.95, label, ha='center', va='center', fontweight='bold')
    
    ax.axhline(y=0.93, xmin=0.03, xmax=0.97, color='black', linewidth=1)
    ax.axhline(y=0.97, xmin=0.03, xmax=0.97, color='black', linewidth=1)
    ax.axhline(y=0.05, xmin=0.03, xmax=0.97, color='black', linewidth=1)

    for i, idx in enumerate(all_indices):
        y_pos = 0.92 - (i + 0.5) * 0.87 / len(all_indices)
        row_height = 0.87 / len(all_indices)
        
        ax.text(cum_widths[0] + column_widths[0]/2, y_pos, f"{i+1}", ha='center', va='center')
        
        pattern = Chem.MolFromSmarts(smarts_patterns[idx])
        substructure_img = Draw.MolToImage(pattern, size=(200, 100))
        img_ax = fig.add_axes([0.15, y_pos - 0.025, 0.15, 0.05])
        img_ax.imshow(substructure_img)
        img_ax.axis('off')
        
        smarts = smarts_patterns[idx]
        if len(smarts) > 20:
            smarts_display = smarts[:17] + "..."
        else:
            smarts_display = smarts

        ax.text(cum_widths[2] + column_widths[2]/2, y_pos, smarts_display, ha='center', va='center')
        ax.text(cum_widths[3] + column_widths[3]/2, y_pos, f"{accuracy_scores[idx]:.3f}", ha='center', va='center')
        ax.text(cum_widths[4] + column_widths[4]/2, y_pos, f"{class_f1[idx]:.3f}", ha='center', va='center')
        ax.text(cum_widths[5] + column_widths[5]/2, y_pos, f"{class_ap[idx]:.3f}", ha='center', va='center')

        num_in_set = int(frequencies[idx])
        ax.text(cum_widths[6] + column_widths[6]/2, y_pos, f"{num_in_set}", ha='center', va='center')
        
        test_count = test_set_counts[idx]
        correct_count = correct_counts[idx]
        ratio = correct_count / test_count if test_count > 0 else 0
        
        ax.text(cum_widths[7] + column_widths[7]/2, y_pos, f"{test_count}", ha='center', va='center')
        ax.text(cum_widths[8] + column_widths[8]/2, y_pos, f"{correct_count}", ha='center', va='center')
        ax.text(cum_widths[9] + column_widths[9]/2, y_pos, f"{ratio:.3f}", ha='center', va='center')

        if i < len(all_indices) - 1:
            y_line = y_pos - row_height/2
            ax.axhline(y=y_line, xmin=0.03, xmax=0.97, color='gray', linewidth=0.5)

    for x in cum_widths[1:]:
        ax.axvline(x=x, ymin=0.05, ymax=0.97, color='black', linewidth=1)
        
    ax.axvline(x=0.03, ymin=0.05, ymax=0.97, color='black', linewidth=1)
    ax.axvline(x=0.97, ymin=0.05, ymax=0.97, color='black', linewidth=1)

    rect = plt.Rectangle((0.03, 0.05), 0.94, 0.88, 
                         facecolor=(0.95, 0.95, 0.95), alpha=0.3, zorder=-1)
    ax.add_patch(rect)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()