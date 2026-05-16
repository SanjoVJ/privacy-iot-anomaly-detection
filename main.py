import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from dp_noise import LaplaceNoise
from stream_loader import load_stream
from evaluator import evaluate_model

# --- SET SEED FOR CONSISTENT RESULTS ---
np.random.seed(42)
random.seed(42)

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "artificialWithAnomaly", "art_daily_jumpsup.csv")

WINDOW_SIZE = 100
epsilons = [0.1, 0.5, 1.0, 5.0]

# 2. Define Models (Adding random_state=42 where applicable)
models_to_test = {
    "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.05, random_state=42),
    "One-Class SVM": OneClassSVM(nu=0.05, kernel="rbf"),
    "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True),
    "Elliptic Envelope": EllipticEnvelope(contamination=0.05, random_state=42)
}

results_private = {name: [] for name in models_to_test}
results_baseline = {}

# 3. Execution
for model_name, model in models_to_test.items():
    print(f"Evaluating Model: {model_name}")
    
    # --- BASELINE ---
    window_values, window_labels, y_true, y_pred = [], [], [], []
    for row in load_stream(file_path):
        window_values.append([row['value']])
        window_labels.append(row['label'])
        if len(window_values) == WINDOW_SIZE:
            X = np.array(window_values)
            model.fit(X)
            preds = [1 if p == -1 else 0 for p in model.predict(X)]
            y_true.extend(window_labels)
            y_pred.extend(preds)
            window_values, window_labels = [], []
    
    _, _, f1_base = evaluate_model(y_true, y_pred)
    results_baseline[model_name] = f1_base
    print(f"Baseline (No Privacy) → F1={f1_base:.3f}")

    # --- PRIVATE ---
    for epsilon in epsilons:
        dp = LaplaceNoise(epsilon=epsilon)
        window_values, window_labels, y_true, y_pred = [], [], [], []
        for row in load_stream(file_path):
            # The noise is random, but np.random.seed makes it the SAME random sequence every time
            noisy_val = dp.add_noise(row['value'])
            window_values.append([noisy_val])
            window_labels.append(row['label'])
            if len(window_values) == WINDOW_SIZE:
                X = np.array(window_values)
                model.fit(X)
                preds = [1 if p == -1 else 0 for p in model.predict(X)]
                y_true.extend(window_labels)
                y_pred.extend(preds)
                window_values, window_labels = [], []
        
        _, _, f1_private = evaluate_model(y_true, y_pred)
        results_private[model_name].append(f1_private)
        print(f"  ε={epsilon} → F1={f1_private:.3f}")
    print("-" * 30)

# 4. Plotting
plt.figure(figsize=(12, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (name, private_scores) in enumerate(results_private.items()):
    color = colors[i]
    plt.plot(epsilons, private_scores, marker='o', label=f"{name} (Private)", color=color, linewidth=2)
    plt.axhline(y=results_baseline[name], color=color, linestyle='--', alpha=0.6, label=f"{name} (Baseline)")

plt.xlabel("Epsilon (Privacy Budget)")
plt.ylabel("F1 Score")
plt.title("Comparison of Anomaly Detection Models: Privacy vs Utility")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()