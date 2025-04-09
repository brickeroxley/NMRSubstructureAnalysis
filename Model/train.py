import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from Model.losses import WeightedBCELoss

# Function which trains the model and evaluates on the validation set
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, weight_decay=1e-5, device='cuda', substructure_frequencies=None):
    model = model.to(device)
    
    # Binary cross-entropy loss function for multi-label classification
    if substructure_frequencies is not None:
        criterion = WeightedBCELoss(substructure_frequencies, device, scaling_factor=3.0)
        print("Using Weighted BCE Loss")
    else:
        criterion = nn.BCELoss()
        print("Using standard BCE Loss")

    # Adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler to reduce learning rate when progress stalls
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ap': [],  
        'val_auc': [],  
        'val_f1': []  
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            h1_spectrum = batch['h1_spectrum'].to(device)
            c13_bins = batch['c13_bins'].to(device)
            formula = batch['formula'].to(device)
            targets = batch['targets'].to(device)
            
            # Zero the gradients, forward pass through model, calculates loss, back-propagates loss to compute gradients, and
            # updates model parameters using optimizer
            optimizer.zero_grad()
            outputs = model(h1_spectrum, c13_bins, formula)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Accumulate weighted loss value 
            train_loss += loss.item() * h1_spectrum.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in val_loader:
                h1_spectrum = batch['h1_spectrum'].to(device)
                c13_bins = batch['c13_bins'].to(device)
                formula = batch['formula'].to(device)
                targets = batch['targets'].to(device)
                
                # Forward pass through model with gradient computation disabled and calculates/accumulates loss
                outputs = model(h1_spectrum, c13_bins, formula)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * h1_spectrum.size(0)

                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate metrics
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        
        # Average precision (AP)
        ap = average_precision_score(all_targets, all_outputs, average='macro')
        history['val_ap'].append(ap)
        
        # Area Under ROC Curve (AUC)
        auc = roc_auc_score(all_targets, all_outputs, average='macro')
        history['val_auc'].append(auc)
        
        # F1 Score
        predictions = (all_outputs > 0.5).astype(np.float32)
        f1 = f1_score(all_targets, predictions, average='macro')
        history['val_f1'].append(f1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"AP: {ap:.4f}, "
              f"AUC: {auc:.4f}, "
              f"F1: {f1:.4f}, "
              f"Time: {epoch_time:.2f}s, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# Evaluates the model on the test set
def evaluate_model(model, test_loader, device, thresholds=None):
    model.eval()
    
    criterion = nn.BCELoss()
    test_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            h1_spectrum = batch['h1_spectrum'].to(device)
            c13_bins = batch['c13_bins'].to(device)
            formula = batch['formula'].to(device)
            targets = batch['targets'].to(device)
            
            # Does a forward pass and calculates loss 
            outputs = model(h1_spectrum, c13_bins, formula)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * h1_spectrum.size(0)
            
            # Calculates predictions and targets
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    # Uses optimized thresholds and evaluation outputs to obtain binary substructure predictions
    if thresholds is not None:
        predictions = np.zeros_like(all_outputs)
        for i in range(all_outputs.shape[1]):
            predictions[:, i] = (all_outputs[:, i] > thresholds[i]).astype(float)
    else:
        predictions = (all_outputs > 0.5).astype(float)
    
    # Calculate metrics
    ap = average_precision_score(all_targets, all_outputs, average='macro')
    auc = roc_auc_score(all_targets, all_outputs, average='macro')
    
    predictions = (all_outputs > 0.5).astype(np.float32)
    f1 = f1_score(all_targets, predictions, average='macro')
    
    print(f"\nTest Set Metrics:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Per-substructure analysis
    class_ap = average_precision_score(all_targets, all_outputs, average=None)
    class_f1 = f1_score(all_targets, predictions, average=None)
    
    return {
        'test_loss': test_loss,
        'ap': ap,
        'auc': auc,
        'f1': f1,
        'class_ap': class_ap,
        'class_f1': class_f1,
        'all_outputs': all_outputs,
        'all_targets': all_targets
    }