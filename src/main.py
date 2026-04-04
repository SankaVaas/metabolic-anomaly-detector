"""
Main entry point for MetabolicAnomalyDetector.
Trains GluFormer model on synthetic data for testing.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys
from src.data.hupa_ucm_loader import HUPAUCMDataLoader
# Add project root to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import SyntheticDataLoader
from src.models.gluformer import GluFormer
from src.utils.metrics import calculate_metrics

def main():
    # Load configuration
    config = load_config()
    print("Config loaded successfully.")
    
    # Use the new real data loader
    data_loader = HUPAUCMDataLoader(data_path="data/HUPA-UCM-Dataset", config=config)
    X, y, sensors = data_loader.load_data()
    print(f"Data shapes: X={X.shape}, y={y.shape}, sensors={sensors.shape}")
    
     # --- SUBSET for fast testing (remove this line for full training) ---
    # Use only 10% of data for quick iteration
    subset_ratio = 0.1  # change to 1.0 for full training
    n_samples = int(X.shape[0] * subset_ratio)
    X = X[:n_samples]
    y = y[:n_samples]
    sensors = sensors[:n_samples]
    print(f"Using subset: {n_samples} samples ({subset_ratio*100}%)")
    # ------------------------------------------------------------------
    # Split data based on config splits
    n_samples = X.shape[0]
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    
    train_end = int(n_samples * train_split)
    val_end = train_end + int(n_samples * val_split)
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    sensors_train = sensors[:train_end]
    sensors_val = sensors[train_end:val_end]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Convert to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    sensors_train_t = torch.FloatTensor(sensors_train).to(device)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    sensors_val_t = torch.FloatTensor(sensors_val).to(device)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t, sensors_train_t)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0,  # set to 0 for Windows to avoid multiprocessing issues
        pin_memory=False
    )
    # Initialize model
    model = GluFormer(config).to(device)
    print(f"Model initialized: {model.__class__.__name__}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    # Training loop (single epoch for testing)
    epochs = 100  # For smoke test, just 1 epoch
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x_batch, y_batch, s_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x_batch, s_batch)  # shape (batch, pred_horizon)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {avg_loss:.4f}")
    
    # Validation evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t, sensors_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        print(f"Validation loss: {val_loss:.4f}")
        
        # Compute metrics on first few samples
        val_preds = val_outputs.cpu().numpy()
        val_true = y_val_t.cpu().numpy()
        metrics = calculate_metrics(val_true.flatten(), val_preds.flatten())
        print("Validation metrics (all horizons):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")
    
    print("Training smoke test completed successfully.")

if __name__ == "__main__":
    main()