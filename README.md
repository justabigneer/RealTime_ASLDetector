Real-Time American Sign Language (ASL) Recognition System
AI & DSML Project

A deep learning–based real-time American Sign Language (ASL) recognition system developed as part of the Artificial Intelligence (AI) and Data Science & Machine Learning (DSML) coursework at Pokhara University.

This project integrates:

 YOLOv8 for hand detection (identification/localization)

 Convolutional Neural Network (CNN) for gesture classification

FastAPI for backend inference

Streamlit for web-based deployment

The system enables real-time recognition of ASL alphabet gestures using a webcam.

📌 Project Motivation

Communication barriers between hearing-impaired individuals and non-signers create accessibility challenges. This project aims to provide a real-time automated solution that detects and recognizes static ASL alphabet gestures using computer vision and deep learning techniques.

The project demonstrates practical application of:

Deep Learning

Computer Vision

Model Deployment

API Integration

Full-stack ML system design

 System Architecture
Webcam Input
      ↓
YOLOv8 Model (Hand Detection)
      ↓
Hand Region Cropping
      ↓
CNN Model (Gesture Classification)
      ↓
FastAPI Backend
      ↓
Streamlit Web Interface
      ↓
Predicted ASL Character

 Model Development
1️⃣ CNN – Gesture Classification

Input Shape: 64 × 64 × 3

Total Output Classes: 29

A–Z

del

space

nothing

Activation Functions:

ReLU (hidden layers)

Softmax (output layer)

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Data Normalization: Pixel scaling (0–1)

Validation Strategy: Stratified K-Fold Cross Validation

2️⃣ YOLOv8 – Hand Detection

Custom-trained YOLOv8 model

Used for real-time hand localization

Cropped bounding box passed to CNN

Confidence threshold configurable

 Dataset Description

Dataset: ASL Alphabet Dataset

Classes: 29

Image Type: RGB

Preprocessing Steps:

Image resizing to 64×64

Normalization

Data augmentation (if applied)

📈 Model Performance
Metric	Value (Replace with yours)
Accuracy	95%
Precision	94%
Recall	93%
F1-Score	94%
ROC-AUC	0.97

Evaluation tools used:

Confusion Matrix

ROC Curve

Accuracy/Loss Graphs

 Deployment Architecture

This system is deployed as a web-based application using a two-layer architecture:

🔹 FastAPI (Backend)

Loads trained CNN and YOLO models

Handles inference requests

Returns JSON prediction responses

Run backend:

uvicorn api.main:app --reload

🔹 Streamlit (Frontend)

User-friendly web interface

Webcam integration

Displays real-time predictions

Run frontend:

streamlit run app/streamlit_app.py

 Project Structure
ASL-Recognition/
│
├── models/
│   ├── cnn_model.keras
│   └── best.pt
│
├── training/
│   ├── cnn_training.py
│   └── yolo_training.py
│
├── app.ph
│── stream_app.py
│
├── requirements.txt
└── README.md

 Technologies Used

Python

TensorFlow / Keras

Ultralytics YOLOv8

OpenCV

FastAPI

Streamlit

NumPy

## Learning Outcomes

Through this AI & DSML project, we gained experience in:

Designing deep learning architectures

Model training and evaluation

Object detection using YOLO

API-based ML deployment

Full-stack ML application development

Performance evaluation using ROC and confusion matrix

👥 Project Team

This project was developed by a team of three students as part of the AI & DSML coursework at Pokhara University.

Aditi kc

Anukriti Thapa

Rashi Bista
