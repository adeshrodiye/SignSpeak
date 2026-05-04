"""
4_evaluate_model.py
====================
Comprehensive model evaluation for research paper / report.

Generates:
  models/confusion_matrix.png
  models/confusion_matrix_normalized.png
  models/roc_curve.png
  models/precision_recall_curve.png
  models/per_class_metrics.png
  models/evaluation_report.txt
  models/evaluation_report.csv

Run AFTER training:
    python 4_evaluate_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from utils import LABELS, FEATURE_SIZE, NUM_CLASSES

CSV_PATH   = os.path.join(BASE, "data",   "keypoints_dataset.csv")
MODEL_PATH = os.path.join(BASE, "models", "model.h5")
OUT_DIR    = os.path.join(BASE, "models")
SEED       = 42
VAL_SPLIT  = 0.15

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset ...")
df = pd.read_csv(CSV_PATH, header=None)
X  = df.iloc[:, :FEATURE_SIZE].values.astype("float32")
y  = df.iloc[:,  FEATURE_SIZE].values.astype("int32")

# Use same split as training so val set is identical
_, X_val, _, y_val = train_test_split(
    X, y, test_size=VAL_SPLIT, stratify=y, random_state=SEED
)
print(f"  Validation samples: {len(X_val)}")

# ── 2. Load model + predict ────────────────────────────────────────────────
print("\n[2/7] Loading model and predicting ...")
model  = tf.keras.models.load_model(MODEL_PATH)
y_prob = model.predict(X_val, verbose=1)   # shape (N, NUM_CLASSES)
y_pred = np.argmax(y_prob, axis=1)
y_true = y_val

# ── 3. Core metrics ────────────────────────────────────────────────────────
print("\n[3/7] Computing core metrics ...")

val_loss, val_acc = model.evaluate(X_val,
    tf.keras.utils.to_categorical(y_true, NUM_CLASSES), verbose=0)

top1_acc  = top_k_accuracy_score(y_true, y_prob, k=1)
top3_acc  = top_k_accuracy_score(y_true, y_prob, k=3)
top5_acc  = top_k_accuracy_score(y_true, y_prob, k=5)
mcc       = matthews_corrcoef(y_true, y_pred)
kappa     = cohen_kappa_score(y_true, y_pred)

report_dict = classification_report(
    y_true, y_pred, target_names=LABELS, output_dict=True
)
macro_p   = report_dict["macro avg"]["precision"]
macro_r   = report_dict["macro avg"]["recall"]
macro_f1  = report_dict["macro avg"]["f1-score"]
weighted_f1 = report_dict["weighted avg"]["f1-score"]

print(f"\n{'='*55}")
print(f"  EVALUATION SUMMARY")
print(f"{'='*55}")
print(f"  Top-1 Accuracy          : {top1_acc*100:.2f}%")
print(f"  Top-3 Accuracy          : {top3_acc*100:.2f}%")
print(f"  Top-5 Accuracy          : {top5_acc*100:.2f}%")
print(f"  Validation Loss         : {val_loss:.4f}")
print(f"  Macro Precision         : {macro_p:.4f}")
print(f"  Macro Recall            : {macro_r:.4f}")
print(f"  Macro F1-Score          : {macro_f1:.4f}")
print(f"  Weighted F1-Score       : {weighted_f1:.4f}")
print(f"  Matthews Corr. Coeff.   : {mcc:.4f}")
print(f"  Cohen's Kappa           : {kappa:.4f}")
print(f"{'='*55}")

# ── 4. Per-class metrics ───────────────────────────────────────────────────
print("\n[4/7] Per-class metrics ...")

per_class = []
for idx, lbl in enumerate(LABELS):
    if lbl not in report_dict:
        continue
    m = report_dict[lbl]
    per_class.append({
        "Label"    : lbl,
        "Precision": round(m["precision"], 4),
        "Recall"   : round(m["recall"],    4),
        "F1-Score" : round(m["f1-score"],  4),
        "Support"  : int(m["support"]),
    })

df_metrics = pd.DataFrame(per_class)
print(df_metrics.to_string(index=False))

# Per-class bar chart
fig, axes = plt.subplots(3, 1, figsize=(16, 14))
x = np.arange(len(df_metrics))
w = 0.28

axes[0].bar(x - w, df_metrics["Precision"], w, label="Precision", color="#4C72B0")
axes[0].bar(x,     df_metrics["Recall"],    w, label="Recall",    color="#55A868")
axes[0].bar(x + w, df_metrics["F1-Score"],  w, label="F1-Score",  color="#C44E52")
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_metrics["Label"], rotation=45, ha="right", fontsize=8)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Per-Class Precision / Recall / F1-Score")
axes[0].legend()
axes[0].axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

# F1 sorted bar
df_sorted = df_metrics.sort_values("F1-Score")
colors = ["#C44E52" if v < 0.85 else "#55A868" for v in df_sorted["F1-Score"]]
axes[1].barh(df_sorted["Label"], df_sorted["F1-Score"], color=colors)
axes[1].set_xlim(0, 1.05)
axes[1].set_title("F1-Score per Class (sorted) — red = below 0.85")
axes[1].axvline(0.85, color="gray", linestyle="--", linewidth=0.8)

# Support (samples per class in val set)
axes[2].bar(df_metrics["Label"], df_metrics["Support"], color="#8172B2")
axes[2].set_xticklabels(df_metrics["Label"], rotation=45, ha="right", fontsize=8)
axes[2].set_title("Validation Samples per Class (Support)")

fig.tight_layout()
path = os.path.join(OUT_DIR, "per_class_metrics.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved -> {path}")

# ── 5. Confusion matrices ──────────────────────────────────────────────────
print("\n[5/7] Confusion matrices ...")

cm = confusion_matrix(y_true, y_pred)

# Raw counts
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS,
            annot_kws={"size": 6}, ax=ax)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual",    fontsize=11)
ax.set_title(f"Confusion Matrix (counts)  —  Acc {top1_acc*100:.2f}%")
ax.tick_params(axis="both", labelsize=7)
fig.tight_layout()
path = os.path.join(OUT_DIR, "confusion_matrix.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved -> {path}")

# Normalized (row = recall per class)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS,
            vmin=0, vmax=1,
            annot_kws={"size": 6}, ax=ax)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual",    fontsize=11)
ax.set_title("Normalized Confusion Matrix (recall per class)")
ax.tick_params(axis="both", labelsize=7)
fig.tight_layout()
path = os.path.join(OUT_DIR, "confusion_matrix_normalized.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved -> {path}")

# ── 6. ROC curves (One-vs-Rest) ────────────────────────────────────────────
print("\n[6/7] ROC curves ...")

y_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))

fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
for i in range(NUM_CLASSES):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

macro_auc = np.mean(list(roc_auc_dict.values()))

fig, ax = plt.subplots(figsize=(10, 8))
# Plot a sample of individual class ROC curves
sample_classes = np.linspace(0, NUM_CLASSES - 1, min(10, NUM_CLASSES), dtype=int)
cmap = plt.cm.get_cmap("tab10", len(sample_classes))
for j, i in enumerate(sample_classes):
    ax.plot(fpr_dict[i], tpr_dict[i], lw=1, alpha=0.7,
            color=cmap(j), label=f"{LABELS[i]} (AUC={roc_auc_dict[i]:.2f})")

ax.plot(fpr_micro, tpr_micro, "k--", lw=2,
        label=f"Micro-avg (AUC={roc_auc_micro:.4f})")
ax.plot([0, 1], [0, 1], "gray", lw=0.8, linestyle=":")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves (One-vs-Rest)  —  Macro AUC = {macro_auc:.4f}")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
path = os.path.join(OUT_DIR, "roc_curve.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved -> {path}")

# ── 7. Precision-Recall curves ─────────────────────────────────────────────
print("\n[7/7] Precision-Recall curves ...")

ap_dict = {}
for i in range(NUM_CLASSES):
    ap_dict[i] = average_precision_score(y_bin[:, i], y_prob[:, i])
mean_ap = np.mean(list(ap_dict.values()))

fig, ax = plt.subplots(figsize=(10, 8))
for j, i in enumerate(sample_classes):
    prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
    ax.plot(rec, prec, lw=1, alpha=0.7,
            color=cmap(j), label=f"{LABELS[i]} (AP={ap_dict[i]:.2f})")

ax.axhline(y=1/NUM_CLASSES, color="gray", linestyle=":", lw=0.8,
           label="Random baseline")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_title(f"Precision-Recall Curves  —  mAP = {mean_ap:.4f}")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
path = os.path.join(OUT_DIR, "precision_recall_curve.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved -> {path}")

# ── Save full text report ──────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  SIGNSPEAKMODEL EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  Dataset         : {len(df)} total samples\n")
    f.write(f"  Validation set  : {len(X_val)} samples\n")
    f.write(f"  Classes         : {NUM_CLASSES}\n\n")
    f.write("-" * 60 + "\n")
    f.write("  OVERALL METRICS\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Top-1 Accuracy          : {top1_acc*100:.2f}%\n")
    f.write(f"  Top-3 Accuracy          : {top3_acc*100:.2f}%\n")
    f.write(f"  Top-5 Accuracy          : {top5_acc*100:.2f}%\n")
    f.write(f"  Validation Loss         : {val_loss:.4f}\n")
    f.write(f"  Macro Precision         : {macro_p:.4f}\n")
    f.write(f"  Macro Recall            : {macro_r:.4f}\n")
    f.write(f"  Macro F1-Score          : {macro_f1:.4f}\n")
    f.write(f"  Weighted F1-Score       : {weighted_f1:.4f}\n")
    f.write(f"  Matthews Corr. Coeff.   : {mcc:.4f}\n")
    f.write(f"  Cohen's Kappa           : {kappa:.4f}\n")
    f.write(f"  Micro-avg AUC (ROC)     : {roc_auc_micro:.4f}\n")
    f.write(f"  Macro-avg AUC (ROC)     : {macro_auc:.4f}\n")
    f.write(f"  Mean Average Precision  : {mean_ap:.4f}\n\n")
    f.write("-" * 60 + "\n")
    f.write("  PER-CLASS REPORT\n")
    f.write("-" * 60 + "\n")
    f.write(classification_report(y_true, y_pred, target_names=LABELS))

print(f"\n  Full report -> {report_path}")

# Save per-class CSV
csv_path = os.path.join(OUT_DIR, "evaluation_report.csv")
df_metrics.to_csv(csv_path, index=False)
print(f"  CSV metrics -> {csv_path}")

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  ALL FILES SAVED TO models/")
print(f"{'='*55}")
print("  confusion_matrix.png")
print("  confusion_matrix_normalized.png")
print("  roc_curve.png")
print("  precision_recall_curve.png")
print("  per_class_metrics.png")
print("  evaluation_report.txt")
print("  evaluation_report.csv")
print(f"{'='*55}\n")
