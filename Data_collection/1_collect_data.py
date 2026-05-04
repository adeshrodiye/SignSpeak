"""
1_collect_data.py
=================
Interactive data collector for sign-language keypoints.

Requires: mediapipe==0.10.9

Controls
--------
  SPACE  -> start / pause recording
  S      -> skip current label
  Q      -> quit and save

Output
------
  data/keypoints_dataset.csv   (174 features + label index per row)
  data/labels.txt
"""

import cv2
import csv
import os
import sys
import time

# ── Mediapipe import with clear error message ──────────────────────────────
try:
    import mediapipe as mp
    mp_hands_mod  = mp.solutions.hands
    mp_face_mod   = mp.solutions.face_mesh
    mp_draw       = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_styles
except AttributeError:
    print("\nERROR: Your mediapipe version is too new and removed 'solutions'.")
    print("Fix:   pip install mediapipe==0.10.9\n")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import LABELS, FEATURE_SIZE, build_feature_vector

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLES_PER_LABEL  = 1000
BASE               = os.path.dirname(os.path.abspath(__file__))
CSV_PATH           = os.path.join(BASE, "data", "keypoints_dataset.csv")
LABELS_PATH        = os.path.join(BASE, "data", "labels.txt")
CAMERA_INDEX       = 0
MIN_DETECT_CONF    = 0.7
MIN_TRACK_CONF     = 0.6

GREEN  = (0, 220, 100)
ORANGE = (0, 165, 255)
RED    = (0, 60, 220)
WHITE  = (255, 255, 255)
DARK   = (30, 30, 30)


def draw_hud(frame, label, count, total, recording, hands_visible, face_visible):
    h, w = frame.shape[:2]

    # Progress bar
    cv2.rectangle(frame, (0, 0), (w, 54), DARK, -1)
    fill = int(w * count / max(total, 1))
    cv2.rectangle(frame, (0, 0), (fill, 54), GREEN if recording else ORANGE, -1)
    cv2.putText(frame, f"Sign: {label}   [{count}/{total}]",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

    state = "REC" if recording else "PAUSED — press SPACE"
    col   = RED if recording else ORANGE
    cv2.putText(frame, state, (w - 300, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

    # Sensor status
    items = [
        ("FACE",   GREEN if face_visible              else RED),
        ("R-HAND", GREEN if hands_visible.get("right") else RED),
        ("L-HAND", GREEN if hands_visible.get("left")  else RED),
    ]
    for i, (txt, c) in enumerate(items):
        x = 10 + i * 130
        cv2.rectangle(frame, (x, h - 38), (x + 118, h - 10), c, -1)
        cv2.putText(frame, txt, (x + 8, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DARK, 2)

    cv2.putText(frame, "SPACE=rec/pause   S=skip   Q=quit",
                (10, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)


def draw_landmarks(frame, hand_results, face_results):
    if hand_results.multi_hand_landmarks:
        for lms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, lms,
                mp_hands_mod.HAND_CONNECTIONS,
                mp_draw_style.get_default_hand_landmarks_style(),
                mp_draw_style.get_default_hand_connections_style(),
            )
    if face_results.multi_face_landmarks:
        mp_draw.draw_landmarks(
            frame,
            face_results.multi_face_landmarks[0],
            mp_face_mod.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(
                color=(180, 180, 180), thickness=1, circle_radius=1
            ),
        )


def main():
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)

    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(LABELS))
    print(f"Labels saved -> {LABELS_PATH}")

    csv_file = open(CSV_PATH, "a", newline="")
    writer   = csv.writer(csv_file)

    hands = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECT_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )
    face_mesh = mp_face_mod.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECT_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}. "
              "Try changing CAMERA_INDEX to 1 or 2.")
        csv_file.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== Sign Language Data Collector ===")
    print(f"Signs     : {LABELS}")
    print(f"Per sign  : {SAMPLES_PER_LABEL} samples")
    print("Controls  : SPACE=start/pause, S=skip, Q=quit\n")

    for label_idx, label in enumerate(LABELS):
        print(f"\n[{label_idx+1}/{len(LABELS)}] Prepare sign: '{label}'  — press SPACE to start")
        count     = 0
        recording = False
        skipped   = False

        while count < SAMPLES_PER_LABEL:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            hand_res = hands.process(rgb)
            face_res = face_mesh.process(rgb)

            rgb.flags.writeable = True

            # Which hands are visible?
            hands_vis = {"right": False, "left": False}
            if hand_res.multi_hand_landmarks:
                for _, hd in zip(hand_res.multi_hand_landmarks,
                                 hand_res.multi_handedness):
                    cam_lbl = hd.classification[0].label
                    if cam_lbl == "Left":
                        hands_vis["right"] = True
                    else:
                        hands_vis["left"] = True

            face_vis = bool(face_res.multi_face_landmarks)

            draw_landmarks(frame, hand_res, face_res)
            draw_hud(frame, label, count, SAMPLES_PER_LABEL,
                     recording, hands_vis, face_vis)

            if recording:
                vec = build_feature_vector(hand_res, face_res, w, h)
                writer.writerow(vec + [label_idx])
                csv_file.flush()
                count += 1

            cv2.imshow("Sign Language Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:              # SPACE
                recording = not recording
                print(f"  {'RECORDING' if recording else 'PAUSED'} — {count} samples so far")
            elif key == ord("s"):
                print(f"  Skipping '{label}' ({count} samples saved)")
                skipped = True
                break
            elif key == ord("q"):
                print("Quitting early.")
                cap.release()
                cv2.destroyAllWindows()
                csv_file.close()
                return

        if not skipped:
            print(f"  Completed '{label}' — {count} samples saved.")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nDone!  Dataset -> {CSV_PATH}")
    print(f"Labels -> {LABELS_PATH}")


if __name__ == "__main__":
    main()