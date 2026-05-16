# Privacy-Preserving Anomaly Detection in IoT Streams

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📌 Overview
This project implements a **lightweight, privacy-preserving anomaly detection framework** designed for real-time IoT data streams in Smart City environments. By integrating **Differential Privacy (DP)** using the Laplace mechanism with the **Isolation Forest** algorithm, the system protects sensitive user data while maintaining high diagnostic utility.

The framework addresses the "Privacy-Utility Tradeoff," ensuring compliance with global regulations like **GDPR** without exceeding the computational limits of edge devices.

## 🚀 Key Features
*   **Differential Privacy:** Implements the Laplace Mechanism ($\epsilon$-differential privacy) to inject controlled noise into raw sensor values.
*   **Streaming Pipeline:** Simulates real-world IoT constraints using a row-by-row streaming loader and a **Sliding Window Buffer ($W=100$)**.
*   **Comparative Analysis:** Evaluates four unsupervised models:
    *   **Isolation Forest (Primary Model)**
    *   One-Class SVM (OC-SVM)
    *   Local Outlier Factor (LOF)
    *   Elliptic Envelope
*   **Optimized for Edge:** Successfully tested on entry-level hardware (Intel Core i3, 4GB RAM).

## 🏗️ System Architecture
The pipeline consists of four modular layers:
1.  **Data Source Layer:** Loads univariate time-series data from the **Numenta Anomaly Benchmark (NAB)**.
2.  **Privacy Preservation Layer:** Applies Laplace noise based on a tunable privacy budget ($\epsilon$).
3.  **Detection & Inference Engine:** Fits unsupervised models on privatized windowed batches.
4.  **Metric Engine:** Calculates Precision, Recall, and F1-score against ground-truth labels (95th percentile thresholding).

## 📊 Results & Analysis
Our experiments identified a **"Sweet Spot" at $\epsilon = 1.0$**. At this level, the Laplace noise acts as a regularizer, allowing the Isolation Forest to achieve an **F1-score of 0.129**, which actually outperformed the non-private baseline (0.104) in specific scenarios.

| Privacy Level | Epsilon ($\epsilon$) | F1-Score (Isolation Forest) |
| :--- | :--- | :--- |
| **Baseline** | No Noise | 0.104 |
| Strong Privacy | 0.1 | 0.060 |
| **Optimal Tradeoff**| **1.0** | **0.129** |
| Weak Privacy | 5.0 | 0.110 |

**Conclusion:** Isolation Forest proved the most resilient to noise compared to boundary-based (OC-SVM) or density-based (LOF) methods due to its tree-based partitioning logic.

## 🛠️ Installation

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the simulation:**
    ```bash
    python main.py
    ```

## 📂 Project Structure
*   `main.py`: The execution engine and model comparison logic.
*   `dp_noise.py`: Implementation of the Laplace DP mechanism.
*   `stream_loader.py`: Streaming simulator with sliding window logic.
*   `evaluator.py`: Metric calculation engine.
*   `data/`: Contains the NAB dataset (e.g., `art_daily_jumpsup.csv`).

## 🔮 Future Work
*   **Federated Learning:** Transitioning to a distributed environment using the Flower framework.
*   **Concept Drift:** Handling evolving data patterns with online retraining.
*   **Edge Optimization:** Deployment on microcontrollers like ESP32 or ARM Cortex-M.
