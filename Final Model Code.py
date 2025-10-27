# ============================================================
# QAE-QAD (PCA + 5000 samples): n_qubits=4, with model save
# ============================================================
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import matplotlib.pyplot as plt
import math, time, os, pickle

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "creditcard.csv"
n_samples_use = 3000       # 使用5000样本
n_qubits = 4                # -> 2^4 = 16 amplitude dimension
n_layers = 2
n_epochs = 10
batch_size = 64
lr = 0.05
random_seed = 42
out_dir = "qae_qad_pca5000_outputs"
os.makedirs(out_dir, exist_ok=True)
np.random.seed(random_seed)

# ---------------------------
# Stage 1: Load & preprocess
# ---------------------------
print("\n【阶段1】数据准备")
data = pd.read_csv(DATA_PATH)
if n_samples_use is not None and n_samples_use < len(data):
    data = data.sample(n=n_samples_use, random_state=random_seed).reset_index(drop=True)

X_all = data.drop(columns=["Class"]).values
y_all = data["Class"].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# PCA 压缩到 2^n_qubits
target_dim = 2 ** n_qubits
pca = PCA(n_components=target_dim, random_state=random_seed)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA → {target_dim}维，累计方差解释率: {np.sum(pca.explained_variance_ratio_):.4f}")

# normalize to unit amplitude
norms = np.linalg.norm(X_pca, axis=1, keepdims=True) + 1e-12
X_amp = X_pca / norms

print(f"Samples used: {len(X_amp)}, amplitude dim: {target_dim}")

# ---------------------------
# Stage 2: QAE circuit
# ---------------------------
print("\n【阶段2】定义量子自编码器电路")
dev = qml.device("default.qubit", wires=n_qubits)

def entangling_layer(nwires):
    for i in range(nwires - 1):
        qml.CNOT(wires=[i, i+1])
    if nwires > 2:
        qml.CNOT(wires=[nwires-1, 0])

@qml.qnode(dev, interface="autograd")
def qae_state(amplitude_vector, enc_params, dec_params):
    qml.AmplitudeEmbedding(features=amplitude_vector, wires=range(n_qubits), pad_with=0.0, normalize=True)
    enc = enc_params.reshape((n_layers, n_qubits))
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(enc[l, i], wires=i)
        entangling_layer(n_qubits)
    dec = dec_params.reshape((n_layers, n_qubits))
    for l in range(n_layers):
        entangling_layer(n_qubits)
        for i in range(n_qubits):
            qml.RY(dec[l, i], wires=i)
    return qml.state()

# ---------------------------
# Stage 3: Loss functions
# ---------------------------
n_enc = n_layers * n_qubits
n_dec = n_layers * n_qubits
flat_param_size = n_enc + n_dec

def fidelity_loss_single(amplitude_vector, flat_params):
    enc = flat_params[:n_enc].reshape((n_layers, n_qubits))
    dec = flat_params[n_enc:].reshape((n_layers, n_qubits))
    amp = np.array(amplitude_vector, dtype=float)
    out_state = qae_state(amp, enc, dec)
    inner = np.sum(np.conjugate(amp) * out_state)
    fid = (np.real(inner) ** 2) + (np.imag(inner) ** 2)
    return 1.0 - fid

def batch_loss_autograd(X_batch, flat_params):
    losses = [fidelity_loss_single(x, flat_params) for x in X_batch]
    return qml.math.mean(qml.math.stack(losses))

def batch_loss_eval(X_batch, flat_params):
    flat = np.array(flat_params, requires_grad=False)
    vals = []
    for x in X_batch:
        v = fidelity_loss_single(x, flat)
        vals.append(np.array(v))
    vals = np.stack(vals).astype(float)
    return float(np.mean(vals))

# ---------------------------
# Stage 4: Train/test split
# ---------------------------
print("\n【阶段4】数据分割")
normal_idx = np.where(y_all == 0)[0]
anomaly_idx = np.where(y_all == 1)[0]
n_train = int(0.8 * len(normal_idx))
train_idx = normal_idx[:n_train]
test_idx = np.concatenate([normal_idx[n_train:], anomaly_idx])

X_train = X_amp[train_idx]
y_train = np.zeros(len(train_idx), dtype=int)
X_test = X_amp[test_idx]
y_test = np.concatenate([
    np.zeros(len(normal_idx) - n_train, dtype=int),
    np.ones(len(anomaly_idx), dtype=int)
])

print(f" Train (normal only): {len(X_train)} samples")
print(f" Test: {len(X_test)} samples (fraud count = {int(np.sum(y_test))})")

# ---------------------------
# Stage 5: Train QAE
# ---------------------------
print("\n【阶段5】训练 QAE (Adam + PCA特征)")
opt = qml.AdamOptimizer(stepsize=lr)
params = np.random.randn(flat_param_size, requires_grad=True)

print(f" Flat param size: {flat_param_size}, Epochs: {n_epochs}, Batch: {batch_size}")
start_time = time.time()
epoch_loss_history = []

for epoch in range(1, n_epochs + 1):
    perm = np.random.permutation(len(X_train))
    epoch_losses = []
    for s in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch}/{n_epochs}", unit="batch"):
        batch_idx = perm[s:s+batch_size]
        X_batch = X_train[batch_idx]
        params = opt.step(lambda p: batch_loss_autograd(X_batch, p), params)
        loss_val = batch_loss_eval(X_batch, params)
        epoch_losses.append(loss_val)
    mean_loss = float(np.mean(epoch_losses))
    epoch_loss_history.append(mean_loss)
    print(f" → Epoch {epoch}/{n_epochs} — Mean batch loss: {mean_loss:.6f}")

total_time = time.time() - start_time
print(f"\n训练完成，总耗时 {total_time:.1f}s")

# ---------------------------
# Stage 6: Evaluate test
# ---------------------------
print("\n【阶段6】测试集评估")
test_losses = []
for x in tqdm(X_test, desc="Scoring test set"):
    test_losses.append(fidelity_loss_single(x, np.array(params, requires_grad=False)))
test_losses = np.array([float(np.array(v)) for v in test_losses])
test_scores = (test_losses - test_losses.min()) / (test_losses.max() - test_losses.min() + 1e-12)

auc = roc_auc_score(y_test, test_scores)
fpr, tpr, thresholds = roc_curve(y_test, test_scores)
opt_idx = np.argmax(tpr - fpr)
threshold = thresholds[opt_idx]
y_pred = (test_scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"\n【QAE Results】")
print(f"AUC: {auc:.4f}, Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
print(f"Confusion: (tn={tn}, fp={fp}, fn={fn}, tp={tp})")

# ---------------------------
# Stage 7: Fraud-only test
# ---------------------------
print("\n【阶段7】纯Fraud数据检测")
fraud_idx = np.where(y_all == 1)[0]
X_fraud = X_amp[fraud_idx]
fraud_losses = [fidelity_loss_single(x, np.array(params, requires_grad=False)) for x in X_fraud]
fraud_losses = np.array([float(np.array(v)) for v in fraud_losses])
fraud_scores = (fraud_losses - test_losses.min()) / (test_losses.max() - test_losses.min() + 1e-12)
fraud_pred = (fraud_scores >= threshold).astype(int)
fraud_rate = fraud_pred.mean() * 100
print(f"Fraud detection rate: {fraud_rate:.2f}%")

# ---------------------------
# Stage 8: IsolationForest (only trained on normal data)
# ---------------------------
print("\n【阶段8】Isolation Forest 基线（仅用正常数据训练）")
iso = IsolationForest(contamination=0.01, random_state=42)
iso.fit(X_train)  # ✅ 改进：只用训练集（正常数据）训练

iso_test_scores = -iso.score_samples(X_test)
iso_test_scores = (iso_test_scores - iso_test_scores.min()) / (iso_test_scores.max() - iso_test_scores.min() + 1e-12)
iso_auc = roc_auc_score(y_test, iso_test_scores)

iso_fraud_scores = -iso.score_samples(X_fraud)
iso_fraud_scores = (iso_fraud_scores - iso_fraud_scores.min()) / (iso_fraud_scores.max() - iso_fraud_scores.min() + 1e-12)
iso_fraud_threshold = np.percentile(iso_fraud_scores, 50)
iso_fraud_pred = (iso_fraud_scores >= iso_fraud_threshold).astype(int)
iso_fraud_rate = iso_fraud_pred.mean() * 100

print(f"IF Test AUC: {iso_auc:.4f}")
print(f"IF Fraud detection rate: {iso_fraud_rate:.2f}%")

# ---------------------------
# Stage 9: Save model (PKL format)
# ---------------------------
print("\n【阶段9】保存模型")
model_data = {
    'qae_params': np.array(params),
    'n_qubits': n_qubits,
    'n_layers': n_layers,
    'pca_model': pca,
    'scaler': scaler,
    'threshold': threshold,
    'test_losses_min': test_losses.min(),
    'test_losses_max': test_losses.max(),
}

model_path = os.path.join(out_dir, "qae_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"✅ QAE模型已保存: {model_path}")

iso_model_path = os.path.join(out_dir, "isolation_forest_model.pkl")
with open(iso_model_path, 'wb') as f:
    pickle.dump(iso, f)
print(f"✅ IsolationForest模型已保存: {iso_model_path}")

# ---------------------------
# Stage 10: Save & visualize
# ---------------------------
print("\n【阶段10】保存结果和可视化")
results = pd.DataFrame({
    "true": y_test,
    "qae_score": test_scores,
    "qae_pred": y_pred,
    "iso_score": iso_test_scores
})
results.to_csv(os.path.join(out_dir, "test_results.csv"), index=False)

pd.DataFrame({"epoch_loss": epoch_loss_history}).to_csv(os.path.join(out_dir, "loss_curve.csv"), index=False)

fraud_results = pd.DataFrame({
    "fraud_qae_score": fraud_scores,
    "fraud_qae_pred": fraud_pred,
    "fraud_iso_score": iso_fraud_scores,
})
fraud_results.to_csv(os.path.join(out_dir, "fraud_results.csv"), index=False)

# Training loss curve
plt.figure(figsize=(7,4))
plt.plot(epoch_loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.title("QAE Training Loss (PCA 5000 samples)")
plt.grid(True)
plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
plt.close()

# Anomaly score distribution
plt.figure(figsize=(8,4))
plt.hist(test_scores[y_test==0], bins=60, alpha=0.6, label="Normal")
plt.hist(test_scores[y_test==1], bins=60, alpha=0.6, label="Fraud")
plt.axvline(threshold, color='r', linestyle='--', label=f"Threshold {threshold:.3f}")
plt.legend()
plt.title("Anomaly Score Distribution (QAE, PCA)")
plt.savefig(os.path.join(out_dir, "score_dist.png"), dpi=150)
plt.close()

# Performance comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ROC curves
fpr_qae, tpr_qae, _ = roc_curve(y_test, test_scores)
fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_test_scores)
axes[0].plot(fpr_qae, tpr_qae, label=f'QAE (AUC={auc:.4f})', linewidth=2)
axes[0].plot(fpr_iso, tpr_iso, label=f'IF (AUC={iso_auc:.4f})', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Fraud detection comparison
methods = ['QAE', 'IF']
fraud_rates = [fraud_rate, iso_fraud_rate]
axes[1].bar(methods, fraud_rates, color=['blue', 'orange'], alpha=0.7)
axes[1].set_ylabel('Fraud Detection Rate (%)')
axes[1].set_title('Fraud Detection Rate Comparison')
axes[1].set_ylim([0, 100])
for i, v in enumerate(fraud_rates):
    axes[1].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=150)
plt.close()

print(f"\n✅ 输出结果保存至: {out_dir}")
print("包含:")
print("  - qae_model.pkl (QAE模型)")
print("  - isolation_forest_model.pkl (IF模型)")
print("  - test_results.csv (测试集结果)")
print("  - fraud_results.csv (Fraud检测结果)")
print("  - loss_curve.csv (训练曲线)")
print("  - training_loss.png (训练损失图)")
print("  - score_dist.png (异常分数分布)")
print("  - comparison.png (性能对比)")