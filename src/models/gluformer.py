"""
GluFormer: Transformer-based model for glucose trajectory forecasting.
Based on decoder-only architecture similar to GPT, but adapted for time-series.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
from .base import BaseModel

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x of shape (batch, seq_len, d_model)."""
        return x + self.pe[:, :x.size(1), :]

class GluFormer(BaseModel):
    """
    Transformer model for glucose forecasting.
    
    Architecture:
        - Linear embedding of input (glucose + optionally sensor features)
        - Positional encoding
        - Transformer encoder (or decoder) stack
        - Prediction head to output future horizon
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.model_config = config.get('model', {})
        self.data_config = config.get('data', {})
        super().__init__(config)
    
    def _build_model(self):
        # Model dimensions
        d_model = self.model_config.get('d_model', 512)
        n_heads = self.model_config.get('n_heads', 8)
        n_layers = self.model_config.get('n_layers', 6)
        dropout = self.model_config.get('dropout', 0.1)
        seq_len = self.data_config.get('window_size', 288)
        pred_horizon = self.data_config.get('prediction_horizon', 12)
        sensor_features = self.model_config.get('sensor_features', 4)   # Now reads from config
        
        # Input embedding: glucose (1) + sensor features
        input_dim = 1 + sensor_features
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head: take the last token's representation (or mean) and project to horizon
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_horizon)
        )
        
        # Store for later use
        self.d_model = d_model
        self.pred_horizon = pred_horizon
        self.seq_len = seq_len
        self.sensor_features = sensor_features
    
    def forward(self, x: torch.Tensor, sensors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Glucose sequence, shape (batch, seq_len)
            sensors: Sensor features, shape (batch, seq_len, n_features). If None, use zeros.
        Returns:
            Predictions of shape (batch, pred_horizon)
        """
        batch, seq_len = x.shape
        if seq_len != self.seq_len:
            # Truncate or pad to expected length (simple: truncate if longer)
            if seq_len > self.seq_len:
                x = x[:, :self.seq_len]
                if sensors is not None:
                    sensors = sensors[:, :self.seq_len, :]
            else:
                # Pad with zeros (not expected during normal use)
                pad = torch.zeros(batch, self.seq_len - seq_len, device=x.device)
                x = torch.cat([x, pad], dim=1)
                if sensors is not None:
                    pad_sensors = torch.zeros(batch, self.seq_len - seq_len, sensors.shape[2], device=x.device)
                    sensors = torch.cat([sensors, pad_sensors], dim=1)
        
        # Combine glucose and sensors
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        if sensors is None:
            sensors = torch.zeros(batch, self.seq_len, 1, device=x.device)  # dummy feature
        # Ensure sensors have correct shape (batch, seq_len, n_features)
        if sensors.dim() == 2:  # if just (batch, seq_len) -> assume 1 feature
            sensors = sensors.unsqueeze(-1)
        combined = torch.cat([x, sensors], dim=-1)  # (batch, seq_len, 1 + n_features)
        
        # Embed
        embedded = self.input_proj(combined)  # (batch, seq_len, d_model)
        pos_encoded = self.pos_encoder(embedded)
        
        # Transformer encoder (no causal mask; we could add later)
        encoded = self.transformer_encoder(pos_encoded)  # (batch, seq_len, d_model)
        
        # Take representation of the last time step (or mean)
        last_token = encoded[:, -1, :]  # (batch, d_model)
        preds = self.pred_head(last_token)  # (batch, pred_horizon)
        return preds

if __name__ == "__main__":
    # Smoke test: instantiate model with a dummy config and run forward pass
    from src.config import load_config
    config = load_config()  # loads from src/config/settings.yaml
    model = GluFormer(config)
    
    batch_size = 4
    seq_len = config['data']['window_size']
    pred_horizon = config['data']['prediction_horizon']
    
    # Generate random glucose (mg/dL) and sensors (HR, HRV, GSR)
    glucose = torch.randn(batch_size, seq_len) * 20 + 100  # ~80-120 mg/dL
    sensors = torch.randn(batch_size, seq_len, 3)  # 3 sensor features
    
    model.eval()
    with torch.no_grad():
        output = model(glucose, sensors)
    
    print(f"GluFormer smoke test passed:")
    print(f"  Input glucose shape: {glucose.shape}")
    print(f"  Output predictions shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, {pred_horizon})")