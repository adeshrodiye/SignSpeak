"""
2_train_model.py
================
Train keypoint-based sign classifier. Exports:
  models/model.h5
  data/labels.txt   (verified)
  models/confusion_matrix.png
  models/training_history.png
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from utils import LABELS, FEATURE_SIZE, NUM_CLASSES

CSV_PATH    = os.path.join(BASE, "data",   "keypoints_dataset.csv")
MODEL_PATH  = os.path.join(BASE, "models", "model.h5")
LABELS_PATH = os.path.join(BASE, "data",   "labels.txt")
CM_PATH     = os.path.join(BASE, "models", "confusion_matrix.png")
HIST_PATH   = os.path.join(BASE, "models", "training_history.png")

os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE, "data"),   exist_ok=True)

BATCH_SIZE  = 64
MAX_EPOCHS  = 200
PATIENCE    = 20
VAL_SPLIT   = 0.15
SEED        = 42
AUG_FACTOR  = 5
JITTER_STD  = 0.015

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── 1. Load ────────────────────────────────────────────────────────────────
print("\n[1/6] Loading dataset ...")
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found. Run 1_collect_data.py first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, header=None)
X  = df.iloc[:, :FEATURE_SIZE].values.astype("float32")
y  = df.iloc[:,  FEATURE_SIZE].values.astype("int32")

print(f"  {len(X)} samples x {FEATURE_SIZE} features, {NUM_CLASSES} classes")
for idx, lbl in enumerate(LABELS):
    print(f"  [{idx:2d}] {lbl:<12s} -> {(y==idx).sum()} samples")

# ── 2. Augment ─────────────────────────────────────────────────────────────
print("\n[2/6] Augmenting ...")

def augment_dataset(X, y, factor=AUG_FACTOR):
    rng   = np.random.RandomState(SEED)
    Xa, ya = [X], [y]

    for i in range(factor):
        Xc = X.copy()

        # Gaussian jitter
        Xc += rng.normal(0, JITTER_STD, Xc.shape).astype("float32")

        # Mirror hands every other iteration
        if i % 2 == 0:
            m = Xc.copy()
            m[:, 0:63:3]   *= -1
            m[:, 63:126:3] *= -1
            Xa.append(m)
            ya.append(y)

        # Small in-plane rotation
        angles = rng.uniform(-np.pi/12, np.pi/12, len(Xc))
        rot = Xc.copy()
        for j, ang in enumerate(angles):
            for sl in [slice(0, 63), slice(63, 126)]:
                v = rot[j, sl].reshape(21, 3)
                c, s = np.cos(ang), np.sin(ang)
                x2 =  c*v[:,0] - s*v[:,1]
                y2 =  s*v[:,0] + c*v[:,1]
                v[:,0], v[:,1] = x2, y2
                rot[j, sl] = v.flatten()
        Xa.append(rot)
        ya.append(y)

        # Landmark dropout
        drop = Xc.copy()
        for j in range(len(drop)):
            idxs = rng.choice(21, size=rng.randint(1, 3), replace=False)
            for ki in idxs:
                drop[j, ki*3:   ki*3+3]    = 0.0
                drop[j, 63+ki*3: 63+ki*3+3] = 0.0
        Xa.append(drop)
        ya.append(y)

    return np.vstack(Xa).astype("float32"), np.concatenate(ya).astype("int32")

X, y = augment_dataset(X, y)
print(f"  After augmentation: {len(X)} samples")

# ── 3. Split ───────────────────────────────────────────────────────────────
print("\n[3/6] Splitting ...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SPLIT, stratify=y, random_state=SEED
)
Y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
Y_val   = tf.keras.utils.to_categorical(y_val,   NUM_CLASSES)
print(f"  Train: {len(X_train)}   Val: {len(X_val)}")

cw_arr = compute_class_weight("balanced",
                               classes=np.arange(NUM_CLASSES), y=y_train)
class_weights = {i: w for i, w in enumerate(cw_arr)}

# ── 4. Model ───────────────────────────────────────────────────────────────
print("\n[4/6] Building model ...")

inputs = keras.Input(shape=(FEATURE_SIZE,))
x = keras.layers.Dense(512)(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.Dropout(0.4)(x)

x = keras.layers.Dense(256)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.Dropout(0.35)(x)

x = keras.layers.Dense(128)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.Dropout(0.3)(x)

x = keras.layers.Dense(64)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs)
model.summary()

lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=30,
    t_mul=2.0,
    m_mul=0.9,
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 5. Train ───────────────────────────────────────────────────────────────
print("\n[5/6] Training ...")
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE,
        restore_best_weights=True, verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=0,
    ),
]

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# ── 6. Evaluate + export ───────────────────────────────────────────────────
print("\n[6/6] Evaluating and saving ...")
val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
print(f"\n  Val accuracy : {val_acc*100:.2f}%")
print(f"  Val loss     : {val_loss:.4f}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print("\n" + classification_report(y_val, y_pred, target_names=LABELS))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
fig.tight_layout()
fig.savefig(CM_PATH, dpi=150)
plt.close(fig)

# History plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"],     label="train")
ax1.plot(history.history["val_accuracy"], label="val")
ax1.set_title("Accuracy"); ax1.legend()
ax2.plot(history.history["loss"],     label="train")
ax2.plot(history.history["val_loss"], label="val")
ax2.set_title("Loss"); ax2.legend()
fig.tight_layout()
fig.savefig(HIST_PATH, dpi=150)
plt.close(fig)

# Save final model + labels
model.save(MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    f.write("\n".join(LABELS))

print(f"\n  model.h5          -> {MODEL_PATH}")
print(f"  labels.txt        -> {LABELS_PATH}")
print(f"  confusion_matrix  -> {CM_PATH}")
print(f"  training_history  -> {HIST_PATH}")
print("\nTraining complete! Run 3_inference.py to test live.")