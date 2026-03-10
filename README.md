# 🧠 Brain Tumor Detection using YOLOv8 with Explainable AI

## 📌 Overview
This project implements an AI-based brain tumor detection system using the YOLOv8 object detection framework.

The model analyzes MRI brain scans and identifies four possible conditions:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

To improve interpretability, the system integrates Explainable AI techniques through heatmap visualization, helping users understand why the model predicted a tumor.

The system also estimates tumor severity using bounding box dimensions extracted from detection results.

---

## ✨ Key Features

### 🔍 Tumor Detection
- YOLOv8 deep learning object detection model
- Automatic tumor localization using bounding boxes
- Multi-class tumor classification
- Detection confidence scoring

---

### 🧠 Explainable AI
- Heatmap overlay highlighting regions of model attention
- Visual explanation of model decisions
- Improves clinical trust and interpretability

---

### 📏 Tumor Severity Estimation
Tumor severity is estimated using bounding box dimensions.

The system computes:
- Tumor width
- Tumor height
- Tumor area

Severity is categorized as:

| Tumor Area (px²) | Severity |
|------------------|----------|
| < 2000 | Small |
| 2000 – 8000 | Medium |
| > 8000 | Large |

---

### 🌐 Interactive Streamlit Interface
The project includes a Streamlit web application for real-time analysis.

Users can:
- Upload MRI scans
- Detect brain tumors automatically
- View detection bounding boxes
- See heatmap explanations
- Adjust detection confidence threshold

---

## 🏗️ Project Architecture
brain-tumor-detection-yolov8
│
├── Brain_tumor.ipynb # Training notebook
├── brain_tumor_app.py # Streamlit web application
├── requirements.txt # Project dependencies
├── README.md # Project documentation
│
├── runs/ # YOLO training outputs
│ └── detect/
│ └── train/
│ └── weights/
│ └── best.pt


---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/Oburah-Peter/brain-tumor-detection-yolov8.git
cd brain-tumor-detection-yolov8


### 2️⃣ Create virtual environment
python -m venv venv


Activate environment:

**Windows**
venv\Scripts\activate
