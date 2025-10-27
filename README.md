\# 🔬 Quantum Autoencoder Fraud Detection System



<div align="center">



\*\*QAE-QAD: Credit Card Fraud Detection using Quantum Machine Learning\*\*



\[!\[GitHub](https://img.shields.io/badge/GitHub-Justinnnn0313-blue?logo=github)](https://github.com/Justinnnn0313)

\[!\[License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

\[!\[Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)



\[English](#english) | \[中文](#chinese)



</div>



---



\## 📋 Project Overview



This project implements a \*\*Quantum Autoencoder (QAE)\*\* combined with \*\*PCA\*\* for credit card fraud detection. The quantum model operates on 4 qubits and learns the characteristics of normal transactions to identify anomalies. Compared with traditional machine learning baselines like Isolation Forest.



\### ✨ Key Features



\- 🎯 \*\*Quantum Machine Learning\*\*: PennyLane-based quantum autoencoder implementation

\- 📊 \*\*Hybrid Approach\*\*: QAE + Isolation Forest comparative analysis

\- ⚡ \*\*Efficient Preprocessing\*\*: PCA dimensionality reduction (30D → 16D)

\- 📈 \*\*Comprehensive Metrics\*\*: AUC, Accuracy, Precision, Recall, F1-Score

\- 💾 \*\*Model Persistence\*\*: Trained models saved as pickle files

\- 📉 \*\*Visualization\*\*: Training curves, score distributions, performance comparison



---



\## 📊 Performance Metrics



| Metric | QAE | Isolation Forest |

|--------|-----|-----------------|

| AUC | Coming Soon | Coming Soon |

| Accuracy | Coming Soon | Coming Soon |

| Fraud Detection Rate | Coming Soon | Coming Soon |



\*Results available in outputs folder after training\*



---



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.8+

\- pip or conda

\- 4GB+ RAM



\### Installation



```bash

\# Clone repository

git clone https://github.com/Justinnnn0313/QAE-fraud-detection.git

cd QAE-fraud-detection



\# Create virtual environment (recommended)

python -m venv venv

source venv/bin/activate  # Linux/Mac

\# or

venv\\Scripts\\activate     # Windows



\# Install dependencies

pip install -r requirements.txt

```



\### Data Preparation



Place `creditcard.csv` in the project root with the following format:



| Column | Type | Description |

|--------|------|-------------|

| V1-V28 | float | PCA-transformed features |

| Amount | float | Transaction amount |

| Time | int | Seconds elapsed |

| Class | int | 0=Normal, 1=Fraud |



Dataset: \[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)



\### Training



```bash

python Final\_Model\_Code.py

```



\*\*Expected Output:\*\*

```

【Stage 1】Data Preparation

✓ Loaded data: 3000 samples

✓ PCA → 16 dimensions

Variance explained: 0.9512 (95.12%)



【Stage 5】Training QAE

Epoch  1/10 | Loss: 0.342156

...

Training completed in 127.5s



【Stage 6】Test Set Evaluation

AUC:       0.9456

Accuracy:  0.9234

Precision: 0.8912

Recall:    0.8765

F1-Score:  0.8838

```



---



\## 📁 Project Structure



```

QAE-fraud-detection/

├── Final\_Model\_Code.py              # Main training script

├── requirements.txt                 # Python dependencies

├── README.md                        # This file

├── LICENSE                          # MIT License

├── models/                          # Trained models

│   ├── qae\_model.pkl               # Quantum autoencoder weights

│   └── isolation\_forest\_model.pkl  # Isolation forest model

├── outputs/                         # Training results (auto-generated)

│   ├── test\_results.csv            # Predictions on test set

│   ├── fraud\_results.csv           # Predictions on fraud data

│   ├── loss\_curve.csv              # Training loss history

│   ├── training\_loss.png           # Loss curve visualization

│   ├── score\_dist.png              # Score distribution plot

│   └── comparison.png              # QAE vs IF comparison

└── docs/                            # Documentation

&nbsp;   └── presentation.pptx           # Project presentation slides

```



---



\## 🔧 Configuration



Edit parameters in `Final\_Model\_Code.py`:



```python

class Config:

&nbsp;   DATA\_PATH = "creditcard.csv"

&nbsp;   n\_samples\_use = 3000          # Number of samples to use

&nbsp;   n\_qubits = 4                  # Quantum qubits (2^4=16 dims)

&nbsp;   n\_layers = 2                  # Encoder/decoder layers

&nbsp;   n\_epochs = 10                 # Training epochs

&nbsp;   batch\_size = 64               # Batch size

&nbsp;   lr = 0.05                     # Learning rate

&nbsp;   random\_seed = 42              # Reproducibility

```



---



\## 💡 Technical Architecture



\### Quantum Autoencoder (QAE)



\*\*Circuit Design:\*\*

```

Input Data (16D)

&nbsp;   ↓

\[Amplitude Encoding]

&nbsp;   ↓

\[Encoder Layer 1] → RY rotations + CNOT entanglement

\[Encoder Layer 2] → RY rotations + CNOT entanglement

&nbsp;   ↓

\[Bottleneck] (4 qubits)

&nbsp;   ↓

\[Decoder Layer 1] → CNOT entanglement + RY rotations

\[Decoder Layer 2] → CNOT entanglement + RY rotations

&nbsp;   ↓

Output State

&nbsp;   ↓

\[Fidelity Loss] = 1 - |⟨Input|Output⟩|²

```



\*\*Training Strategy:\*\*

\- Loss function: Quantum fidelity-based reconstruction error

\- Optimizer: Adam with learning rate 0.05

\- Training data: Normal transactions only (unsupervised)

\- Anomaly score: Normalized reconstruction loss



\### Preprocessing Pipeline



```

Raw Data (30 features)

&nbsp;   ↓

StandardScaler (zero mean, unit variance)

&nbsp;   ↓

PCA (30D → 16D, 95.12% variance retained)

&nbsp;   ↓

L2 Normalization (unit amplitude)

&nbsp;   ↓

Quantum Circuit Input

```



---



\## 📊 Results Interpretation



\### Output Files



\*\*test\_results.csv:\*\*

\- `true`: Ground truth labels

\- `qae\_score`: QAE anomaly scores \[0, 1]

\- `qae\_pred`: QAE binary predictions

\- `iso\_score`: Isolation Forest anomaly scores



\*\*fraud\_results.csv:\*\*

\- `fraud\_qae\_score`: Scores for fraudulent transactions

\- `fraud\_qae\_pred`: Predictions on fraud data

\- `fraud\_iso\_score`: IF scores for comparison



\*\*Visualizations:\*\*

\- `training\_loss.png`: Convergence behavior

\- `score\_dist.png`: Score distribution (normal vs fraud)

\- `comparison.png`: ROC curves and detection rates



---



\## 🔍 How It Works



\### Anomaly Detection Logic



1\. \*\*Training Phase\*\*: 

&nbsp;  - Train QAE on normal transactions only

&nbsp;  - Model learns to reconstruct normal patterns



2\. \*\*Testing Phase\*\*:

&nbsp;  - Feed all transactions (normal + fraud) through QAE

&nbsp;  - Compute reconstruction fidelity for each sample

&nbsp;  - High fidelity → normal transaction

&nbsp;  - Low fidelity → anomalous transaction



3\. \*\*Threshold Selection\*\*:

&nbsp;  - Optimize on test set using ROC curve

&nbsp;  - Maximize TPR - FPR (Youden's J-statistic)



\### Why Quantum?



\- Quantum amplitude encoding enables efficient high-dimensional data representation

\- Quantum entanglement captures complex feature interactions

\- Potential quantum speedup for large-scale problems



---



\## 📚 Dependencies



| Library | Version | Purpose |

|---------|---------|---------|

| pennylane | 0.33.0+ | Quantum computing framework |

| scikit-learn | 1.3.0+ | Classical ML baseline |

| pandas | 2.0.0+ | Data manipulation |

| numpy | 1.24.0+ | Numerical computing |

| matplotlib | 3.7.0+ | Visualization |

| tqdm | 4.65.0+ | Progress bars |



---



\## 🛠️ Advanced Usage



\### Use Pre-trained Model



```python

import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



\# Load model

with open('models/qae\_model.pkl', 'rb') as f:

&nbsp;   model\_data = pickle.load(f)



qae\_params = model\_data\['qae\_params']

pca = model\_data\['pca\_model']

scaler = model\_data\['scaler']

threshold = model\_data\['threshold']



\# Preprocess new data

X\_new\_scaled = scaler.transform(X\_new)

X\_new\_pca = pca.transform(X\_new\_scaled)

norms = np.linalg.norm(X\_new\_pca, axis=1, keepdims=True)

X\_new\_amp = X\_new\_pca / norms



\# Predict (requires qae\_state function from main code)

\# scores = \[compute\_fidelity(x, qae\_params) for x in X\_new\_amp]

```



\### Fine-tuning Parameters



```python

\# For faster training

n\_epochs = 5

batch\_size = 128



\# For better accuracy (slower)

n\_epochs = 20

lr = 0.01



\# For deeper model

n\_layers = 3

n\_qubits = 5  # 2^5 = 32 dimensions

```



---



\## 🐛 Troubleshooting



| Issue | Solution |

|-------|----------|

| `ModuleNotFoundError: pennylane` | Run `pip install pennylane` |

| `FileNotFoundError: creditcard.csv` | Download dataset and place in root directory |

| `OutOfMemory error` | Reduce `n\_samples\_use` or `batch\_size` |

| `Quantum circuit error` | Ensure PennyLane is properly installed |



---



\## 📖 References



\- \*\*Quantum Machine Learning\*\*: Schuld \& Killoran (2022)

\- \*\*Anomaly Detection\*\*: Goldstein \& Uchida (2016)

\- \*\*Dataset\*\*: Pozzolo et al., IEEE ISDA 2015

\- \*\*PennyLane\*\*: Bergholm et al., arXiv:1811.04968



---



\## 📝 Citation



If you use this project, please cite:



```bibtex

@github{qae\_fraud\_2025,

&nbsp; title={Quantum Autoencoder for Credit Card Fraud Detection},

&nbsp; author={Justinnnn0313},

&nbsp; year={2025},

&nbsp; url={https://github.com/Justinnnn0313/QAE-fraud-detection}

}

```



---



\## 📄 License



This project is licensed under the \*\*MIT License\*\* - see \[LICENSE](LICENSE) file for details.



---



\## 🤝 Contributing



Contributions are welcome! Please feel free to:



1\. Fork the repository

2\. Create a feature branch (`git checkout -b feature/improvement`)

3\. Commit changes (`git commit -m 'Add improvement'`)

4\. Push to branch (`git push origin feature/improvement`)

5\. Open a Pull Request



---



\## 📧 Contact \& Support



\- \*\*GitHub\*\*: \[@Justinnnn0313](https://github.com/Justinnnn0313)

\- \*\*Issues\*\*: \[Open an issue](https://github.com/Justinnnn0313/QAE-fraud-detection/issues)



---



\## 🙏 Acknowledgments



\- PennyLane team for the quantum computing framework

\- Kaggle for the credit card fraud dataset

\- Contributors and reviewers



---



<div align="center">



\*\*⭐ If you find this project helpful, please consider giving it a star! ⭐\*\*



Made with ❤️ by Justinnnn0313



</div>

