"""
Data loader for HUPA-UCM dataset using preprocessed CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler

class HUPAUCMDataLoader:
    def __init__(self, data_path: str, config: dict):
        self.data_path = Path(data_path)
        self.config = config
        self.sample_rate_minutes = config['data'].get('sample_rate_minutes', 5)
        self.window_size = config['data'].get('window_size', 48)   # 4 hours
        self.pred_horizon = config['data'].get('prediction_horizon', 12)  # 1 hour
        self.glucose_scaler = StandardScaler()
        self.sensor_scaler = StandardScaler()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess all patient CSV files from the Preprocessed folder.
        Returns:
            X: glucose sequences (n_samples, window_size)
            y: target glucose (n_samples, pred_horizon)
            sensors: sensor data (n_samples, window_size, n_features)
        """
        preprocessed_dir = self.data_path / "Preprocessed"
        csv_files = list(preprocessed_dir.glob("HUPA*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No HUPA*.csv files found in {preprocessed_dir}")

        all_glucose = []
        all_sensors = []
        all_targets = []

        for file in csv_files:
            # Read CSV with semicolon separator
            df = pd.read_csv(file, sep=';', parse_dates=['time'])
            df.set_index('time', inplace=True)
            # Resample to fixed interval (e.g., 5 minutes)
            df = df.resample(f'{self.sample_rate_minutes}T').mean().interpolate()
            # Drop rows with missing glucose
            df = df.dropna(subset=['glucose'])
            
            glucose = df['glucose'].values.astype(np.float32)
            # Use heart_rate, steps, calories as sensor features
            sensors = df[['heart_rate', 'steps', 'calories']].values.astype(np.float32)
            
            # Create sliding windows
            X_p, y_p, s_p = self._create_sequences(glucose, sensors)
            if len(X_p) > 0:
                all_glucose.append(X_p)
                all_sensors.append(s_p)
                all_targets.append(y_p)

        if not all_glucose:
            raise ValueError("No valid sequences created from any patient file.")

        X = np.concatenate(all_glucose, axis=0)
        y = np.concatenate(all_targets, axis=0)
        sensors = np.concatenate(all_sensors, axis=0)

        # Normalize
        X_flat = X.reshape(-1, 1)
        X_norm = self.glucose_scaler.fit_transform(X_flat).reshape(X.shape)
        sensors_flat = sensors.reshape(-1, sensors.shape[-1])
        sensors_norm = self.sensor_scaler.fit_transform(sensors_flat).reshape(sensors.shape)

        return X_norm, y, sensors_norm

    def _create_sequences(self, glucose: np.ndarray, sensors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y, s = [], [], []
        total_len = len(glucose)
        for i in range(total_len - self.window_size - self.pred_horizon):
            X.append(glucose[i:i+self.window_size])
            y.append(glucose[i+self.window_size:i+self.window_size+self.pred_horizon])
            s.append(sensors[i:i+self.window_size])
        return np.array(X), np.array(y), np.array(s)