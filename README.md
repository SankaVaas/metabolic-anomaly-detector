# MetabolicAnomalyDetector

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Generative foundation model for early diabetic detection using multi-modal non-invasive sensing.**

This repository implements **GluFormer** – a transformer-based architecture that learns the language of glucose dynamics to detect metabolic anomalies via **digital twin discordance analysis**, without traditional blood pricks.

## 🚀 Key Features

- **Non-invasive multi-modal sensing** – PPG, GSR, and optional Raman spectroscopy
- **Generative forecasting** – Autoregressive transformer predicts future glucose trajectories
- **Digital twin discordance** – Detects metabolic maladaptation by comparing predicted vs. actual physiology
- **Privacy-preserving** – Federated learning architecture (ready for deployment)
- **Synthetic data generator** – For prototyping and testing

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/metabolic-anomaly-detector.git
cd metabolic-anomaly-detector

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .