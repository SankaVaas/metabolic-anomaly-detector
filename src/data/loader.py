"""
Synthetic data generator for MetabolicAnomalyDetector
Generates realistic simulated glucose and sensor data for testing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

def generate_synthetic_glucose(
    n_samples: int = 10000,
    sample_rate_minutes: int = 10,
    include_anomalies: bool = True,
    anomaly_ratio: float = 0.1,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic glucose time series with realistic patterns.
    
    The signal combines:
    - Basal circadian rhythm (24h period)
    - Postprandial spikes (3 meals/day)
    - Random physiological noise
    - Optional anomalies (simulating diabetic events)
    
    Returns:
        Array of shape (n_samples,) glucose values in mg/dL
    """
    np.random.seed(random_seed)
    
    t = np.arange(n_samples) * sample_rate_minutes / 60.0  # hours
    
    # Circadian rhythm: ~24h period, amplitude ~20 mg/dL, baseline ~90
    circadian = 20 * np.sin(2 * np.pi * t / 24.0)
    
    # Meal spikes: three peaks per day, modelled as Gaussian bumps
    meals = np.zeros_like(t)
    meal_times = np.arange(7, 24, 8)  # 7am, 3pm, 11pm (simplified)
    for meal_hour in meal_times:
        # Meal effect centered around meal_hour, width 1h, amplitude 30-50 mg/dL
        peak = 40 * np.exp(-0.5 * ((t - meal_hour) / 1.0) ** 2)
        meals += peak
    
    # Noise (physiological variability)
    noise = 3 * np.random.randn(n_samples)
    
    # Combine
    glucose = 90 + circadian + meals + noise
    
    # Clamp to realistic range [50, 300]
    glucose = np.clip(glucose, 50, 300)
    
    if include_anomalies:
        # Replace some segments with anomalous values (diabetic spikes or drops)
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        # Anomaly types: high spike (200-300) or low (50-70)
        spike_mask = np.random.rand(n_anomalies) > 0.5
        glucose[anomaly_indices[spike_mask]] = np.random.uniform(200, 300, np.sum(spike_mask))
        glucose[anomaly_indices[~spike_mask]] = np.random.uniform(50, 70, np.sum(~spike_mask))
    
    return glucose.astype(np.float32)

def generate_synthetic_sensors(
    n_samples: int = 10000,
    sample_rate_minutes: int = 10,
    include_artifacts: bool = True,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic sensor data (PPG, GSR) correlated with glucose.
    
    PPG features: heart rate variability metrics
    GSR: skin conductance correlated with stress (which correlates with glucose)
    
    Returns:
        DataFrame with columns: hr, hrv, gsr, motion_artifact (if include_artifacts)
    """
    np.random.seed(random_seed)
    
    glucose = generate_synthetic_glucose(n_samples, sample_rate_minutes, include_anomalies=False, random_seed=random_seed)
    
    # Simulate heart rate (HR) – inverse correlation with glucose? Actually HR increases with stress/high glucose.
    # We'll make HR positively correlated with glucose with noise.
    hr_baseline = 70
    hr = hr_baseline + 0.2 * (glucose - 90) + 5 * np.random.randn(n_samples)
    hr = np.clip(hr, 50, 120)
    
    # Heart rate variability (HRV) – decreases with stress.
    hrv_baseline = 50
    hrv = hrv_baseline - 0.1 * (glucose - 90) + 5 * np.random.randn(n_samples)
    hrv = np.clip(hrv, 20, 80)
    
    # GSR (skin conductance) – increases with arousal/stress.
    gsr_baseline = 2.0
    gsr = gsr_baseline + 0.03 * (glucose - 90) + 0.5 * np.random.randn(n_samples)
    gsr = np.clip(gsr, 1.0, 6.0)
    
    data = pd.DataFrame({
        'hr': hr,
        'hrv': hrv,
        'gsr': gsr,
    })
    
    if include_artifacts:
        # Simulate motion artifacts: random spikes in HR and GSR
        artifact_ratio = 0.02
        n_artifacts = int(n_samples * artifact_ratio)
        artifact_idx = np.random.choice(n_samples, n_artifacts, replace=False)
        data.loc[artifact_idx, 'hr'] += np.random.uniform(20, 50, n_artifacts)
        data.loc[artifact_idx, 'gsr'] += np.random.uniform(2, 5, n_artifacts)
        data['motion_artifact'] = 0
        data.loc[artifact_idx, 'motion_artifact'] = 1
    
    return data

class SyntheticDataLoader:
    """Simple synthetic data loader that returns glucose and sensor data."""
    
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config['data']['sample_rate_minutes']
        self.window_size = config['data']['window_size']
        self.pred_horizon = config['data']['prediction_horizon']
        self.train_split = config['data']['train_split']
        self.val_split = config['data']['val_split']
        self.test_split = config['data']['test_split']
        
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load synthetic dataset.
        Returns:
            X: glucose time series (n_samples,)
            y: future glucose values (n_samples,)
            sensors: sensor data (n_samples, n_features)
        """
        # Generate longer sequence to allow windows
        total_len = self.window_size + self.pred_horizon + 1000  # extra buffer
        glucose = generate_synthetic_glucose(
            n_samples=total_len,
            sample_rate_minutes=self.sample_rate,
            include_anomalies=True,
            random_seed=42
        )
        sensors_df = generate_synthetic_sensors(
            n_samples=total_len,
            sample_rate_minutes=self.sample_rate,
            include_artifacts=True,
            random_seed=42
        )
        
        # Convert to sliding windows
        X, y, sensors = [], [], []
        for i in range(total_len - self.window_size - self.pred_horizon):
            X.append(glucose[i:i+self.window_size])
            y.append(glucose[i+self.window_size:i+self.window_size+self.pred_horizon])
            sensors.append(sensors_df.iloc[i:i+self.window_size].values)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        sensors = np.array(sensors, dtype=np.float32)
        
        return X, y, sensors

if __name__ == "__main__":
    # Quick smoke test
    config = {
        'data': {
            'sample_rate_minutes': 10,
            'window_size': 288,
            'prediction_horizon': 12,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        }
    }
    loader = SyntheticDataLoader(config)
    X, y, sensors = loader.load()
    print(f"Synthetic data generated successfully:")
    print(f"  X shape: {X.shape} (samples, window_size)")
    print(f"  y shape: {y.shape} (samples, horizon)")
    print(f"  sensors shape: {sensors.shape} (samples, window_size, features)")
    print(f"  Glucose range: [{X.min():.1f}, {X.max():.1f}] mg/dL")