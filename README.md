# 🤟 SignSpeak — AI Hand Sign Recognition: Convert Hand Signs into Real-Time Subtitles

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Django](https://img.shields.io/badge/Django-Backend-green?logo=django)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmark%20Detection-red)
![WebRTC](https://img.shields.io/badge/WebRTC-P2P%20Video-blueviolet)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Real-time Indian Sign Language (ISL) detection using MediaPipe keypoint extraction, a custom TensorFlow neural network, and a full-stack Django web platform with peer-to-peer video calling.**

[Features](#-features) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack) • [Getting Started](#-getting-started) • [Project Structure](#-project-structure) • [Team](#-team)

</div>

---

## 📌 Overview

SignSpeak bridges the communication gap between deaf and hearing individuals by converting Indian Sign Language hand gestures into real-time text subtitles and spoken audio. It runs entirely through a standard web browser with a webcam — no specialized hardware required.

> 🏫 Final Year BE (AI & Data Science) Project — PVG's College of Engineering, Technology & Management, Pune (Savitribai Phule Pune University) | AY 2025–26

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖐️ **Real-Time Gesture Recognition** | Detects ISL hand signs at 25–30 FPS using webcam |
| 📝 **Live Subtitle Generation** | Converts recognized gestures into text captions instantly |
| 🔊 **Text-to-Speech** | Converts generated captions to audio via gTTS |
| 💡 **Word Suggestions** | Context-aware next-word prediction to speed up communication |
| 📹 **P2P Video Calling** | Peer-to-peer video calls with caption overlay via WebRTC (MiroTalk C2C) |
| 🔐 **User Authentication** | Secure login/signup with OTP support via Django |
| 🌗 **Dark / Light Theme** | Responsive UI with theme switching |
| 🌐 **Browser-Based** | No installation needed for end users — works in Chrome/Edge |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│              Webcam / Video Call Stream                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    PROCESSING LAYER                         │
│  Video Capture → Hand Detection (MediaPipe) →               │
│  Gesture Recognition Model (TensorFlow DNN) →               │
│  Gesture-to-Text Mapping                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  APPLICATION LAYER                          │
│  Subtitle Generator → User Interface → Subtitle Logger (DB) │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  INTERACTION LAYER                          │
│     Video Call Adapter / Virtual Camera API → WebRTC        │
└─────────────────────────────────────────────────────────────┘
```

The ML pipeline (data collection, model training, and live inference) lives in the **`Data_collection/`** subfolder of this repository. The web platform (Django + WebRTC) is in the **`Hand-Sign-Module/`** subfolder.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Hand Landmark Detection** | MediaPipe (21 landmarks/hand + 24 face keypoints → 174-D feature vector) |
| **Gesture Classification** | TensorFlow — Residual Deep Neural Network (DNN) |
| **Video Processing** | OpenCV |
| **Backend** | Django (REST APIs, authentication, streaming) |
| **Frontend** | HTML5, CSS3, JavaScript (Responsive UI) |
| **Database** | SQLite (via Django ORM) |
| **Video Calling** | WebRTC — MiroTalk C2C |
| **Text-to-Speech** | gTTS (Google Text-to-Speech) / Browser Speech API |
| **Deployment** | Nginx (reverse proxy) + PM2 (process manager) |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure

```
SignSpeak/
│
├── Data_collection/               # ML Pipeline (see its own README)
│   ├── 0_check_dataset.py         # Inspect collected dataset
│   ├── 0_remove_sign_data.py      # Remove unwanted gesture data
│   ├── 1_collect_data.py          # Capture keypoints via webcam
│   ├── 2_train_model.py           # Train TensorFlow DNN model
│   ├── 3_interface.py             # Run standalone inference interface
│   ├── 4_evaluate_model.py        # Evaluate accuracy & confusion matrix
│   ├── utils.py                   # Shared utility functions
│   ├── requirements.txt           # ML-specific dependencies
│   └── README.md                  # Data collection & training guide
│
├── Hand-Sign-Module/              # Django Web Application
│   ├── app/                       # Core Django app (views, models, URLs)
│   ├── global_project/            # Django project settings
│   ├── model/                     # Trained model files (model.h5, labels.txt)
│   ├── static/                    # CSS, JS, images
│   ├── templates/                 # HTML templates
│   ├── manage.py                  # Django management script
│   └── requirements.txt           # Web app dependencies
│
└── README.md                      # ← You are here
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or above
- Webcam (720p or higher recommended)
- Modern browser (Chrome / Edge) with WebRTC support
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/adeshrodiye/SignSpeak.git
cd SignSpeak
```

### 2. Set Up the Web Application

```bash
cd Hand-Sign-Module
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open your browser and navigate to `http://127.0.0.1:8000`

### 3. Train or Use Your Own Model

See [`Data_collection/README.md`](./Data_collection/README.md) for the complete guide on collecting gesture data and training the model.

### Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i3 | Intel i5 / i7 or equivalent |
| RAM | 4 GB | 8 GB+ |
| GPU | — | NVIDIA RTX 3060 (for training) |
| Webcam | 720p | 1080p |

---

## 🧠 How It Works

1. **Capture** — Webcam stream is captured frame-by-frame via OpenCV.
2. **Detect** — MediaPipe extracts 21 hand landmarks per hand and 24 facial keypoints, forming a 174-dimensional normalized feature vector.
3. **Classify** — The feature vector is fed into a trained Residual DNN (TensorFlow) which outputs the predicted gesture label.
4. **Smooth** — A rolling window mechanism stabilizes predictions across frames.
5. **Output** — Recognized gestures are accumulated into words and sentences, displayed as subtitles, and optionally spoken aloud via TTS.

---

## 📊 Performance

| Metric | Value |
|---|---|
| Model Accuracy | ~99.99% (on training dataset — see confusion matrix) |
| Target FPS | 25–30 FPS |
| Caption Latency | 400–600 ms |
| Feature Vector Size | 174 dimensions |
| Signs Supported | 26+ ISL gestures |

---

## 🔮 Future Work

- Continuous sentence-level sign language recognition (not just isolated words)
- Mobile application (Android / iOS)
- Multi-language subtitle support
- Improved dataset diversity for better generalization across users
- Two-way communication with speech-to-sign translation

---

## 📄 Publication

The project was presented at the **3rd International Conference on Advances in Engineering and Medical Sciences 2026** (15–18 April 2026), organized by the International School of Technology and Sciences for Women (Autonomous), India.

> **Paper Title:** *SignSpeak: Real-Time Sign Language Recognition, Caption Generation and Peer-to-Peer Video Communication Platform*

---

## 👨‍💻 Team

| Name | Role | Exam No |
|---|---|---|
| **Aniket Raskar** | Team Lead & Dataset Engineer | B400070464 |
| **Adesh Rodiye** | ML Model Developer | B400070465 |
| **Shreyash Patkar** | Backend Developer | B400070463 |
| **Mayur Bari** | Frontend & Integration Developer | B400070413 |

**Guide:** Prof. Pallavi G. Bangale, Dept. of AI & DS  
**Institution:** PVG's College of Engineering, Technology & Management, Pune — 411009

---

## 📜 License

This project was developed as an academic final year project submitted to Savitribai Phule Pune University. All work is original and certified by the institute.
