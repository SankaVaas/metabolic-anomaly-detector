"""
Evaluation metrics for glucose forecasting and anomaly detection.
"""

import numpy as np
from typing import Dict, Union, List

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%)."""
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if np.sum(mask) == 0:
        return 0.0
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def calculate_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List]
) -> Dict[str, float]:
    """
    Calculate standard regression metrics for glucose prediction.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with keys: MAE, RMSE, MAPE
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    assert y_true.shape == y_pred.shape, "Shapes of true and predicted values must match"

    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }

if __name__ == "__main__":
    # Smoke test: simple example
    y_true = np.array([100, 110, 120, 130])
    y_pred = np.array([102, 108, 118, 133])
    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics smoke test:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    # Test individual functions
    print(f"\nMAE: {mae(y_true, y_pred):.2f}")
    print(f"RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")