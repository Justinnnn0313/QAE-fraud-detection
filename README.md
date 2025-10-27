# 🔬 Quantum Autoencoder Fraud Detection (QAE-QAD)

**Credit Card Fraud Detection using Quantum Machine Learning**
---

## 🌐 Language
[English](#english) | [中文](#中文版本)

---

## 🧭 Overview

This project implements a **Quantum Autoencoder (QAE)** combined with **PCA** for credit card fraud detection.  
The model learns the internal structure of **normal transactions** to identify **fraudulent anomalies**.  
A classical baseline (Isolation Forest) is included for performance comparison.

---

## 💡 Motivation

Traditional machine learning models often struggle with **high-dimensional, sparse, and nonlinear** credit card transaction data.  
Quantum methods, however, naturally operate in **exponentially large Hilbert spaces**, allowing them to represent and entangle complex feature interactions efficiently.

In this project:
- The **Quantum Autoencoder** encodes transaction data into a smaller latent space while preserving meaningful quantum correlations.
- This **expressive quantum representation** enables better separation between normal and fraudulent patterns in high-dimensional feature spaces.

> ⚛️ *In short: Quantum circuits may capture subtler statistical dependencies that classical models miss.*

---

## ⚙️ Key Features

- 🧠 **Quantum Machine Learning** — Implemented via [PennyLane](https://pennylane.ai/)
- 🔗 **Hybrid Architecture** — QAE + PCA + Isolation Forest comparison
- ⚡ **Dimensionality Reduction** — PCA (30 → 16 dimensions)
- 📈 **Comprehensive Metrics** — AUC, Accuracy, Precision, Recall, F1
- 💾 **Model Persistence** — Trained models saved as `.pkl`
- 📉 **Visualization** — Training curves, ROC comparison, score distributions

---

## 📊 Performance (Sample Results)

|     Metric    | Quantum Autoencoder | Isolation Forest |
|---------------|---------------------|------------------|
|    **AUC**    |        0.9456       |       0.8731     |
|  **Accuracy** |        0.9234       |       0.8910     |
| **Precision** |        0.8912       |       0.8534     |
|   **Recall**  |        0.8765       |       0.8012     |
| **F1-Score**  |        0.8838       |       0.8265     |

> 💡 *Results generated on 3,000 sampled transactions. Full results available in `/outputs`.*

---

## 🚀 Quick Start

### 1️⃣ Setup Environment
```bash
git clone https://github.com/Justinnnn0313/QAE-fraud-detection.git
cd QAE-fraud-detection

python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
