🧠 ISL Medical Sign Recognition using 5-Layer CNN

This project implements a deep learning–based real-time Indian Sign Language (ISL) interpreter for medical-related terms using a custom 5-layer Convolutional Neural Network (CNN). It supports recognition of 7 common clinical gestures such as `cold`, `cough`, `fever`, and `pain`, and is designed to assist communication between hearing-impaired individuals and healthcare professionals.

📌 Features

- ✅ Recognizes 7 medical-related ISL gestures
- 🎥 Real-time webcam-based prediction with confidence overlay
- 📊 Evaluation with accuracy, F1-score, confusion matrix, reliability diagrams
- 🧠 Trained on custom video dataset using frame-based CNN
- 🧼 Includes preprocessing, training, evaluation, and real-time scripts

🗂️ Project Structure

```bash
cnn_isl/
├── data/
│   ├── raw/                # Original videos
│   ├── processed/          # Numpy arrays (X_train.npy, etc.) [ignored]
├── models/                 # Trained model files (.h5) [ignored]
├── src/
│   ├── preprocess.py       # Video frame extraction + processing
│   ├── train.py            # CNN model training
│   ├── evaluate.py         # Confusion matrix + metrics
│   ├── real_time_predict.py# Real-time webcam prediction
├── README.md
├── .gitignore
├── requirements.txt
└── config.py

