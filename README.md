<div align="center">

#  Multimodal Federated Learning for Smart City Flood Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

**A resilient, privacy-preserving, and computationally efficient disaster early warning system.**

</div>

---

## Overview

This project implements a **Heterogeneous Federated Learning (HeteroFL)** architecture designed for real-time flood detection in smart city environments. Traditional flood warning systems rely heavily on centralized data collection, which raises privacy concerns, suffers from communication bottlenecks, and is prone to single points of failure. 

Our **TriModalFloodNet** model overcomes these challenges by processing and fusing data locally on edge devices (like IoT sensors and surveillance cameras) and only sharing learned model weights.

By simultaneously ingesting **Satellite Imagery**, **River Gauge (Water Level) Data**, and **Rainfall Time-Series**, the system achieves exceptional predictive performance and resilience, even when individual sensor modalities drop out.

---

## Key Innovations

*   **Heterogeneous Federated Learning (HeteroFL):** Not all edge devices are created equal. Our architecture dynamically slices the global model into smaller sub-networks (e.g., `Rate 0.5` for low-power edge nodes, `Rate 1.0` for HQ servers), allowing computationally constrained IoT devices to participate in the federated training process.
*   **Deep Trimodal Fusion:** 
    *   **Image Branch:** A pre-trained [ResNet-18](https://arxiv.org/abs/1512.03385) backbone (with unfrozen high-level layers) and CBAM-style Spatial Attention for satellite imagery.
    *   **Rainfall Branch:** An all-MLP [TSMixer](https://arxiv.org/abs/2303.06053) optimized for multivariate time-series forecasting.
    *   **Water-Level Branch:** A [NARX](https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model) (Nonlinear AutoRegressive Network with eXogenous inputs) architecture for hydrological sequence modeling.
*   **Modality Dropout & Sensor Resilience:** We implement aggressive Modality Dropout (dropping sensor data 50% of the time during training) to prevent "sensor dominance," ensuring the model learns strong representations from the visual data branch as well.
*   **Recall-Optimized Focal Loss:** In disaster management, a false negative (missed flood) is far more dangerous than a false positive. We utilize a Focal Loss function with optimized decision thresholds to heavily penalize missed detections.
*   **Monte Carlo Uncertainty Quantification:** Deep stochastic layers allow the network to output not just a binary classification, but a calibrated *predictive uncertainty variance*, providing critical trust metrics for emergency responders.

---

## System Architecture

The core architecture (`models/fusion.py`) operates as follows:

1. Modality-specific encoders extract high-level feature representations.
2. The representations are z-score normalized and mapped to a unified feature space.
3. The features are concatenated and passed through a Deep Stochastic Fusion Head with dual MC-Dropout layers.
4. The global server aggregates these weights using HeteroFL aggregation rules (`heterofl/heterofl_ops.py`), intelligently masking and averaging weights based on the active channel capacities of participating clients.

---

##  Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is highly recommended to use an Anaconda environment.

```bash
conda create -n flood_env python=3.10
conda activate flood_env
```

### 2. Dependencies
Core dependencies required for this project:
- **PyTorch & Torchvision:** Core deep learning framework and pre-trained ResNet.
- **Scikit-learn:** Advanced evaluation metrics (ROC-AUC, PR-AUC).
- **Pandas & NumPy:** Time-series data processing and synthetic sensor data generation.
- **Matplotlib:** Generating publication-ready performance charts.

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib
```

### 3. Dataset Information & Structure
Place your data into a `dataset/` folder at the root of the project:
*   `dataset/flood_dataset_npy/`: Directory containing 64x64 `.npy` satellite patches. **These patches are sourced from the [Sen1Floods11 dataset](https://github.com/cloudtostreet/Sen1Floods11).**
*   `dataset/flood_dataset.csv`: Water-level sequence data.
*   `dataset/tsmixer_input.csv`: Rainfall time-series data.

*(Note: Due to the scarcity of paired multimodal disaster datasets, the time-series data for the rainfall and river gauge sensors were synthetically generated using custom `numpy` distributions to simulate realistic physical correlations with the satellite imagery labels. The actual dataset files are excluded from this repository via `.gitignore` due to size constraints.)*

---

## Usage

To launch the federated training pipeline across all simulated edge clients:

```bash
python main.py
```

**The pipeline will automatically:**
1. Slice the model for different edge devices (Satellite Node, River Gauge Node, HQ Server).
2. Train the subnetworks locally using the simulated clients.
3. Aggregate the updates at the global server.
4. Perform an Ablation Study (evaluating unimodal performance).
5. Generate Uncertainty Histograms using Monte Carlo Dropout.
6. Save the final optimized weights to `flood_model.pth`.

---

## Performance & Results

The repository includes a plotting script to generate performance charts:

```bash
python codes/plots.py
```
*Charts will be saved to the `figures/` directory.*

### Evaluation Metrics (Held-out Test Set)
*   **F1-Score:** ~94%
*   **Nash-Sutcliffe Efficiency (NSE):** >0.90
*   **System Resiliency:** Maintains >85% F1-score even under 30% channel dropout (missing sensor data).

---

## Repository Structure

```text
├── heterofl/                  # Core Federated Learning Logic
│   ├── config.py              # Hyperparameters and client configurations
│   ├── data_loader.py         # Multi-modal alignment and padding
│   ├── heterofl_ops.py        # Model slicing and federated aggregation
│   ├── losses.py              # Recall-optimized Focal Loss
│   ├── mc_dropout.py          # Uncertainty quantification and histograms
│   ├── plotting.py            # Learning curve visualization
│   └── training.py            # Local training loops and global evaluation
├── models/                    # Neural Network Architectures
│   ├── cnn.py                 # DoubleConv & Spatial Attention blocks
│   ├── fusion.py              # TriModalFloodNet global model
│   ├── narx.py                # Water-level autoregressive feature extractor
│   └── tsmixer.py             # Rainfall time-series MLPs
├── codes/                     # Utility Scripts
│   ├── generate_journal_plots.py
│   └── graph_gen.py
├── main.py                    # Pipeline execution entry point
├── .gitignore                 # Protected data ignore rules
└── README.md                  # Project documentation
```

---

##  Contributing
Contributions to improve the feature extractors, add new IoT sensor modalities, or optimize the HeteroFL aggregation protocols are welcome. Please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.