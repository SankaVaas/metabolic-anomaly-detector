"""
Visualization utilities for glucose forecasting and anomaly detection.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, List, Tuple

def plot_glucose_trajectory(
    time_series: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    true_future: Optional[np.ndarray] = None,
    title: str = "Glucose Trajectory",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot glucose time series with optional predictions and true future values.
    
    Args:
        time_series: Historical glucose values (shape: seq_len)
        predictions: Predicted future values (shape: horizon)
        true_future: Actual future values (shape: horizon)
        title: Plot title
        save_path: If provided, save plot to this path
        show: If True, display plot (default: True)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    x_hist = np.arange(len(time_series))
    plt.plot(x_hist, time_series, 'b-', label='Historical', linewidth=2)
    
    # Plot predictions and true future if provided
    if predictions is not None:
        x_pred = np.arange(len(time_series), len(time_series) + len(predictions))
        plt.plot(x_pred, predictions, 'r--', label='Predicted', linewidth=2)
    
    if true_future is not None:
        x_true = np.arange(len(time_series), len(time_series) + len(true_future))
        plt.plot(x_true, true_future, 'g-', label='Actual', linewidth=2)
    
    plt.xlabel('Time steps')
    plt.ylabel('Glucose (mg/dL)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_error_distribution(
    errors: np.ndarray,
    title: str = "Prediction Error Distribution",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot histogram of prediction errors.
    
    Args:
        errors: Array of errors (actual - predicted)
        title: Plot title
        save_path: If provided, save plot to this path
        show: If True, display plot
    """
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Zero error')
    plt.xlabel('Error (mg/dL)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_sensor_correlations(
    glucose: np.ndarray,
    sensors: np.ndarray,
    sensor_names: List[str],
    title: str = "Glucose vs Sensor Features",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Scatter plot of glucose vs each sensor feature.
    
    Args:
        glucose: Glucose values (shape: n_samples)
        sensors: Sensor values (shape: n_samples, n_features)
        sensor_names: Names of sensor features
        title: Overall plot title
        save_path: If provided, save plot to this path
        show: If True, display plot
    """
    n_features = sensors.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
    if n_features == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, sensor_names)):
        ax.scatter(sensors[:, i], glucose, alpha=0.5, s=1)
        ax.set_xlabel(name)
        ax.set_ylabel('Glucose (mg/dL)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # Smoke test: generate dummy data and create plots
    np.random.seed(42)
    n_steps = 200
    glucose = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, n_steps)) + 5 * np.random.randn(n_steps)
    pred_horizon = 20
    predictions = glucose[-pred_horizon:] + np.random.randn(pred_horizon) * 3
    true_future = glucose[-pred_horizon:]
    errors = true_future - predictions
    os.makedirs('tests/smoke_tests', exist_ok=True)
    save_folder='tests/smoke_tests'
    # Test plot_glucose_trajectory (save to file to avoid display)
    plot_glucose_trajectory(glucose[:-pred_horizon], predictions, true_future,
                            title="Smoke Test: Glucose Forecast",
                            save_path=f"{save_folder}/smoke_test_glucose.png", show=False)
    
    # Test error distribution
    plot_error_distribution(errors, save_path=f"{save_folder}/smoke_test_errors.png", show=False)
    
    # Test sensor correlations (dummy sensors)
    sensors = np.random.randn(n_steps, 2) + 0.1 * glucose.reshape(-1, 1)
    plot_sensor_correlations(glucose, sensors, ['HR', 'GSR'],
                            save_path=f"{save_folder}/smoke_test_correlations.png", show=False)
    
    print("Visualization smoke test passed: plots saved as smoke_test_*.png")