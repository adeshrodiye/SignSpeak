"""
utils.py  —  Shared keypoint normalization helpers.

Compatible with mediapipe 0.10.9 (solutions API).

Feature vector layout (per frame):
  [0  :  63]  right hand  — 21 landmarks x (x, y, z)  — zeros if absent
  [63 : 126]  left  hand  — 21 landmarks x (x, y, z)  — zeros if absent
  [126: 174]  face        — 24 key landmarks x (x, y)
  Total = 174 features
"""

import numpy as np

# ── Safe mediapipe import ──────────────────────────────────────────────────
try:
    import mediapipe as mp
    _test = mp.solutions.hands   # will raise AttributeError on wrong version
except AttributeError:
    raise ImportError(
        "\n\nYour mediapipe version does not have the 'solutions' API.\n"
        "Run:  pip install mediapipe==0.10.9\n"
    )

# ── Hand normalization ─────────────────────────────────────────────────────

def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    span = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts /= span
    return pts.flatten().tolist()

def hand_zeros():
    return [0.0] * 63

# ── Face normalization ─────────────────────────────────────────────────────

FACE_KEY_IDS = [
    33,  133,
    362, 263,
    1,
    4,
    61,  291,
    0,   17,
    152,
    10,
    234, 454,
    70,  300,
    105, 334,
    6,   168,
    195, 5,
    98,  327,
]
assert len(FACE_KEY_IDS) == 24

def normalize_face(landmarks, img_w, img_h):
    pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in FACE_KEY_IDS],
        dtype=np.float32,
    )
    pts -= pts.mean(axis=0)
    span = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts /= span
    return pts.flatten().tolist()

def face_zeros():
    return [0.0] * 48

# ── Feature vector assembly ────────────────────────────────────────────────

FEATURE_SIZE = 63 + 63 + 48   # = 174

def build_feature_vector(hand_results, face_results, img_w, img_h):
    right_vec = hand_zeros()
    left_vec  = hand_zeros()

    if hand_results.multi_hand_landmarks:
        for hand_lms, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness,
        ):
            cam_label = handedness.classification[0].label
            vec = normalize_hand(hand_lms.landmark)
            if cam_label == "Left":
                right_vec = vec
            else:
                left_vec = vec

    if face_results.multi_face_landmarks:
        face_vec = normalize_face(
            face_results.multi_face_landmarks[0].landmark, img_w, img_h
        )
    else:
        face_vec = face_zeros()

    return right_vec + left_vec + face_vec

# ── Labels ─────────────────────────────────────────────────────────────────
# IMPORTANT: Never insert labels in the middle — always append to the end.
# Existing label indices (0-25) must remain unchanged to keep old CSV data valid.

LABELS = [
    # ── Original labels (indices 0–25) — DO NOT reorder ──
    "HELLO",
    "THANKS",
    "TEACHER",
    "INDIAN",
    "I_AM",
    "YOU_ARE",
    "BEAUTIFUL",
    "GOOD",
    "PRACTICE",
    "MAN",
    "WOMAN",
    "PLACE",
    "TIME",
    "MARRY",
    "HOUSE",
    "FOOD",
    "0", 
    "1", 
    "2", 
    "3", 
    "4", 
    "5", 
    "6", 
    "7", 
    "8", 
    "9",  
    # ── A–Z letters (indices 26–49) ──
    # Note: J and Z involve motion gestures — collect carefully or skip for now.
    # Confusable groups needing 700–800 samples each:
    #   A / E / S  (closed fist variants, differ in thumb position)
    #   M / N / T  (fingers folded over thumb, differ slightly)
    #   R / U / V  (two-finger signs, differ in spread/cross)
    "A", 
    "B", 
    "C", 
    "D", 
    "E", 
    "F", 
    "G", 
    "H", 
    "I",
    "J",   # motion sign — collect carefully (hook motion)
    "K", 
    "L", 
    "M", 
    "N", 
    "O", 
    "P", 
    "Q",
    "R",
    "S", 
    "T",
    "U", 
    "2/V",  #it has same sign as 2 
    "W", 
    "X", 
    "Y",
    "Z",   # motion sign — collect carefully (zigzag motion)     jb

    # ── Safety class (index 52) — catches ambiguous/no-sign poses ──
    # Collect ~400 samples of random neutral hand positions for this.
    # "UNKNOWN",
]

NUM_CLASSES = len(LABELS)