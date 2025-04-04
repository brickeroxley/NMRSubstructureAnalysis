import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import h5py

from Model.dataset import NMRSubstructureDataset
from Model.model import NMRSubstructurePredictor
from Model.train import train_model, evaluate_model
from Model.visualization import plot_training_history, visualize_predictions, analyze_feature_importance, create_performance_table
from Model.utils import set_seed, save_config, count_parameters, optimize_thresholds

def main():
    # Configuration for file paths, substructure amount, and ML hyperparameters
    config = {
        'h1_file': "Simulated_Spectra/simulation_out/1H_processed/preprocessed_h_nmr_data.h5",
        'c13_file': "Simulated_Spectra/simulation_out/13C_processed/preprocessed_c_nmr_data.h5",
        'formula_file': "Molecular Formulas/hdf5/molecular_formulas.h5",
        'substructure_file': "Substructures/substructure_analysis.h5",
        'num_substructures': 422,
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'seed': 8,
        'train_size': 0.8,
        'val_size': 0.1,
        'test_size': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save configuration of ML parameters
    save_config(config, 'results/config.json')
    
    # Set seed for reproducibility 
    set_seed(config['seed'])
    
    print(f"Using device: {config['device']}")
    
    print(f"Loading datasets...")
    dataset = NMRSubstructureDataset(
        config['h1_file'],
        config['c13_file'],
        config['formula_file'],
        config['substructure_file']
    )
    
    # Split dataset based on ratios detailed in the configuration dictionary
    # Train: 80%, Validation: 10%, Test: 10% 
    train_size = int(config['train_size'] * len(dataset))
    val_size = int(config['val_size'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset splits: Train = {train_size}, Validation = {val_size}, Test = {test_size}")
    
    # Creating PyTorch DataLoader objects for each dataset 
    # Training data is shuffled as to not introduce any ordering bias in the dataset
    # Multiple workers are used to enable multi-processing to boost performance
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model
    model = NMRSubstructurePredictor(num_substructures=config['num_substructures'])
    model.initialize_weights()
    
    # Number of trainable parameters in the model
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Train model
    print(f"Starting training...")
    start_time = time.time()

    with h5py.File(config['substructure_file'], 'r') as f:
        substructure_frequencies = f['substructure_frequencies'][:]
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device'],
        substructure_frequencies=substructure_frequencies
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save trained model for later retrieval 
    torch.save(trained_model.state_dict(), 'models/nmr_substructure_predictor.pth')
    
    # Plot training history
    plot_training_history(history)
    
    # Find optimal thresholds on validation set
    print("Optimizing prediction thresholds...")
    optimal_thresholds = optimize_thresholds(
        trained_model, 
        val_loader, 
        config['device'], 
        config['num_substructures']
    )

    # Evaluate with optimized thresholds
    print("Evaluating on test set with optimized thresholds...")
    test_results = evaluate_model(
        trained_model, 
        test_loader, 
        config['device'], 
        thresholds=optimal_thresholds
    )

    # Creates a performance table to visualize certain substructure prediction performance
    visualize_performance_table = True
    if visualize_performance_table:
        create_performance_table(
            test_results,
            config['substructure_file'],
            save_path='results/substructure_performance_table.png'
    )
    
    # Find best and worst performing substructures
    best_indices = np.argsort(test_results['class_ap'])[-10:]
    worst_indices = np.argsort(test_results['class_ap'])[:10]
    
    print("\nTop 10 Best Predicted Substructures:")
    for idx in best_indices[::-1]:
        print(f"Substructure {idx}: AP = {test_results['class_ap'][idx]:.4f}, F1 = {test_results['class_f1'][idx]:.4f}")
    
    print("\nTop 10 Worst Predicted Substructures:")
    for idx in worst_indices:
        print(f"Substructure {idx}: AP = {test_results['class_ap'][idx]:.4f}, F1 = {test_results['class_f1'][idx]:.4f}")
    
    # Visualize predictions on a few test examples
    sample_indices = list(range(0, len(test_dataset), len(test_dataset)//10))[:5]
    actual_indices = [test_dataset.indices[i] for i in sample_indices]
    
    visualize_predictions(
        trained_model, 
        dataset, 
        actual_indices, 
        config['device'], 
        config['substructure_file']
    )
    
    # Analyze feature importance
    analyze_feature_importance(trained_model, test_dataset, config['device'])
    
    print("Job's done")

if __name__ == "__main__":
    main()