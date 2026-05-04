"""
3_interface.py
==============
Real-time sign recognition using model.h5.

Controls:  Q = quit   R = reset vote history
"""

import cv2
import numpy as np
import os
import sys
import time
import collections
from collections import Counter

import tensorflow as tf

try:
    import mediapipe as mp
    mp_hands_mod  = mp.solutions.hands
    mp_face_mod   = mp.solutions.face_mesh
    mp_draw       = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles
except AttributeError:
    print("\nERROR: mediapipe 'solutions' API missing.")
    print("Fix:   pip install mediapipe==0.10.9\n")
    sys.exit(1)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from utils import LABELS, FEATURE_SIZE, build_feature_vector

MODEL_PATH   = os.path.join(BASE, "models", "model.h5")
LABELS_PATH  = os.path.join(BASE, "data",   "labels.txt")
CAMERA_INDEX = 0

# ── Thresholds ─────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.88       # raised from 0.75 — reduces false predictions
VOTE_WINDOW    = 12         # slightly wider window for stability
MIN_DETECT     = 0.7
MIN_TRACK      = 0.6

# ── Display map ────────────────────────────────────────────────────────────
# Maps internal label names -> what is shown on screen.
# Add any label here if you want a custom display string.
DISPLAY_MAP = {
    # Words
    "HELLO"     : "HELLO",
    "THANKS"    : "THANKS",
    "TEACHER"   : "TEACHER",
    "INDIAN"    : "INDIAN",
    "I_AM"      : "I AM",
    "YOU_ARE"   : "YOU ARE",
    "BEAUTIFUL" : "BEAUTIFUL",
    "GOOD"      : "GOOD",
    "PRACTICE"  : "PRACTICE",
    "MAN"       : "MAN",
    "WOMAN"     : "WOMAN",
    "PLACE"     : "PLACE",
    "TIME"      : "TIME",
    "MARRY"     : "MARRY",
    "HOUSE"     : "HOUSE",
    "FOOD"      : "FOOD",

    # Numbers
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",

    # Letters — ambiguous pairs shown with context
    "A" : "A",
    "B" : "B",
    "C" : "C",
    "D" : "D",
    "E" : "E",
    "F" : "F",
    "G" : "G",
    "H" : "H",
    "I" : "I",
    "J" : "J",
    "K" : "K",
    "L" : "L",
    "M" : "M",
    "N" : "N",
    "O" : "O",
    "P" : "P",
    "Q" : "Q",
    "R" : "R",
    "S" : "S",
    "T" : "T",
    "U" : "U",
    "2/V" : "2 / V",    # shown with context since 2 and V look similar
    "W" : "6",    # shown with context since 6 and W look similar
    "X" : "X",
    "Y" : "Y",
    "Z" : "Z",

    # Safety class — show nothing when pose is ambiguous
    "UNKNOWN"   : "...",
}

GREEN  = (60,  220, 80)
ORANGE = (0,   165, 255)
RED    = (60,  60,  220)
WHITE  = (255, 255, 255)
DARK   = (25,  25,  25)
HAND_R = (50,  200, 255)
HAND_L = (255, 180, 50)
FACE_C = (180, 180, 180)


def draw_hand_landmarks(frame, hand_results):
    if not hand_results.multi_hand_landmarks:
        return
    for lms, hd in zip(hand_results.multi_hand_landmarks,
                        hand_results.multi_handedness):
        col  = HAND_R if hd.classification[0].label == "Left" else HAND_L
        spec = mp_draw.DrawingSpec(color=col, thickness=2, circle_radius=3)
        mp_draw.draw_landmarks(
            frame, lms, mp_hands_mod.HAND_CONNECTIONS, spec, spec)


def draw_face_landmarks(frame, face_results):
    if not face_results.multi_face_landmarks:
        return
    mp_draw.draw_landmarks(
        frame,
        face_results.multi_face_landmarks[0],
        mp_face_mod.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=FACE_C, thickness=1, circle_radius=1),
    )


def draw_panel(frame, label_text, confidence, n_votes):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), DARK, -1)
    cv2.putText(frame, label_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, GREEN, 3)
    bar_w = int((w - 280) * min(confidence, 1.0))
    cv2.rectangle(frame, (250, 32), (w - 20, 60), (60, 60, 60), -1)
    cv2.rectangle(frame, (250, 32), (250 + bar_w, 60), GREEN, -1)
    cv2.putText(frame, f"{confidence*100:.0f}%",
                (250, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    cv2.putText(frame, f"votes {n_votes}/{VOTE_WINDOW}",
                (w - 160, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1)
    cv2.putText(frame, "Q=quit  R=reset", (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1)


def draw_status(frame, hand_count, face_visible):
    h, w = frame.shape[:2]
    items = [
        (f"HANDS:{hand_count}", GREEN if hand_count > 0 else RED),
        ("FACE",                GREEN if face_visible    else RED),
    ]
    for i, (txt, col) in enumerate(items):
        x = w - 230 + i * 115
        cv2.rectangle(frame, (x, h - 36), (x + 106, h - 10), col, -1)
        cv2.putText(frame, txt, (x + 5, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK, 2)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model.h5 not found at {MODEL_PATH}")
        print("Run 2_train_model.py first.")
        sys.exit(1)

    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = open(LABELS_PATH).read().splitlines()
    print(f"Model loaded. {len(labels)} labels.")

    hands = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECT,
        min_tracking_confidence=MIN_TRACK,
    )
    face_mesh = mp_face_mod.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECT,
        min_tracking_confidence=MIN_TRACK,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    votes = collections.deque(maxlen=VOTE_WINDOW)
    confs = collections.deque(maxlen=VOTE_WINDOW)

    print("Running — Q to quit, R to reset.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_res = hands.process(rgb)
        face_res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        # ── Fix: no prediction when hands not visible ──────────────────────
        hand_count = len(hand_res.multi_hand_landmarks) \
                     if hand_res.multi_hand_landmarks else 0

        if hand_count == 0:
            # Don't predict — show dash instead of wrong label like WOMAN
            display   = "—"
            best_conf = 0.0
            draw_face_landmarks(frame, face_res)
            draw_hand_landmarks(frame, hand_res)
            draw_panel(frame, display, best_conf, len(votes))
            draw_status(frame, hand_count, bool(face_res.multi_face_landmarks))
            cv2.imshow("Sign Language Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                votes.clear(); confs.clear()
                print("History reset.")
            continue

        # ── Predict ────────────────────────────────────────────────────────
        vec  = build_feature_vector(hand_res, face_res, w, h)
        prob = model.predict(np.array([vec], dtype="float32"), verbose=0)[0]
        idx  = int(np.argmax(prob))
        conf = float(prob[idx])

        votes.append(idx)
        confs.append(conf)

        best_idx  = Counter(votes).most_common(1)[0][0]
        best_conf = float(np.mean([confs[i] for i, v
                                   in enumerate(votes) if v == best_idx]))

        raw = labels[best_idx] if best_idx < len(labels) else "?"

        if best_conf >= CONF_THRESHOLD:
            display = DISPLAY_MAP.get(raw, raw.upper())
            # Hide UNKNOWN class predictions from screen
            if raw == "UNKNOWN":
                display, best_conf = "...", 0.0
        else:
            display, best_conf = "...", 0.0

        draw_face_landmarks(frame, face_res)
        draw_hand_landmarks(frame, hand_res)
        draw_panel(frame, display, best_conf, len(votes))
        draw_status(frame, hand_count, bool(face_res.multi_face_landmarks))

        cv2.imshow("Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            votes.clear(); confs.clear()
            print("History reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()