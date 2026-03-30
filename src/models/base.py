"""
Base model class for MetabolicAnomalyDetector
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in MetabolicAnomalyDetector.
    
    All models should inherit from this class and implement:
    - forward()
    - configure_optimizers()
    - compute_loss()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.
        
        Args:
            config: Model configuration dictionary (e.g., from settings.yaml['model'])
        """
        super().__init__()
        self.config = config
        self._build_model()
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture (must be overridden)."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, sensors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input glucose sequence [batch_size, seq_len]
            sensors: Optional sensor data [batch_size, seq_len, n_features]
        
        Returns:
            Predictions [batch_size, pred_horizon]
        """
        pass
    
    def configure_optimizers(self, learning_rate: Optional[float] = None):
        """
        Configure optimizer and optionally scheduler.
        
        Args:
            learning_rate: Override default from config if provided
        
        Returns:
            torch.optim.Optimizer
        """
        lr = learning_rate or self.config.get('learning_rate', 1e-4)
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.config.get('weight_decay', 0.0))
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between predictions and targets.
        
        Args:
            pred: Model predictions [batch_size, pred_horizon]
            target: Ground truth [batch_size, pred_horizon]
        
        Returns:
            Scalar loss
        """
        return nn.MSELoss()(pred, target)
    
    def predict_step(self, x: torch.Tensor, sensors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Wrapper for forward() used during inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, sensors)

if __name__ == "__main__":
    # Quick smoke test – check if abstract class can be instantiated with a dummy implementation
    class DummyModel(BaseModel):
        def _build_model(self):
            self.fc = nn.Linear(self.config['d_model'], self.config.get('pred_horizon', 12))
        
        def forward(self, x, sensors=None):
            # x shape: [batch, seq_len]
            # For dummy, we just take the last value and pass through linear
            last_val = x[:, -1:]  # [batch, 1]
            return self.fc(last_val)
    
    config = {'d_model': 512, 'pred_horizon': 12}
    model = DummyModel(config)
    dummy_input = torch.randn(4, 288)  # batch=4, seq_len=288
    output = model(dummy_input)
    print(f"BaseModel smoke test passed: output shape {output.shape}")
    print(f"Expected: [4, 12]")