# Hybrid-CNN-SVM-Image-Forgery-Detection
# 🔐 ForgeGuard — Hybrid AI Document Forgery Detection

Hybrid CNN + Classical Machine Learning system for detecting forged documents.

Built using MobileNetV2 deep feature embeddings combined with handcrafted texture features and an RBF-SVM classifier.

Achieved **90% Accuracy** and **0.958 ROC-AUC** on 13,000 document images.

---

## 🚀 Overview

ForgeGuard is a hybrid document forgery detection system that combines:

- Deep transfer learning (MobileNetV2 embeddings)
- Classical image texture analysis (LBP, HOG, DCT, Edge Density, JPEG artifacts)
- PCA dimensionality reduction
- RBF Support Vector Machine classifier
- Threshold optimization for improved decision boundaries
- FastAPI backend
- Flask frontend interface

This hybrid approach significantly improves performance over classical-only methods.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 90% |
| ROC-AUC | 0.958 |
| Precision (avg) | 0.90 |
| Recall (avg) | 0.90 |

Balanced dataset:
- 6,500 Original
- 6,500 Forged

---

## 🧠 Architecture

Frontend (Flask)
↓  
FastAPI Backend  
↓  
Hybrid Feature Extraction  
&nbsp;&nbsp;&nbsp;&nbsp;• MobileNetV2 Embeddings  
&nbsp;&nbsp;&nbsp;&nbsp;• LBP  
&nbsp;&nbsp;&nbsp;&nbsp;• HOG  
&nbsp;&nbsp;&nbsp;&nbsp;• DCT Statistics  
&nbsp;&nbsp;&nbsp;&nbsp;• Edge Density  
&nbsp;&nbsp;&nbsp;&nbsp;• JPEG Artifact Score  
↓  
PCA  
↓  
RBF-SVM  
↓  
Optimized Threshold Classification  

---

## 🗂 Project Structure
DocumentForgeryDetection/
│
├── backend.py
├── app.py
├── forgery_detection_fast.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best_model.joblib
│
├── templates/
│   └── index.html
│
├── results/
│   ├── roc.png
│   ├── cm.png
│
└── .gitignore

---

## ⚙ Installation

Create virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate

Install dependencies:
pip install tensorflow fastapi uvicorn flask opencv-python numpy scikit-learn scikit-image scipy matplotlib seaborn joblib

🏋️ Train the Model
python forgery_detection_fast.py

This will:

Extract deep + classical features

Train hybrid SVM model

Optimize threshold

Save best model to /models

🌐 Run the Backend
uvicorn backend:app --reload

Backend runs at:

http://127.0.0.1:8000

🖥 Run the Frontend

In another terminal:

python app.py

Open:

http://127.0.0.1:5000

Upload document images and analyze forgery probability.

🔍 Key Innovations

Hybrid feature learning (deep + handcrafted)

Dimensionality reduction for stability

Decision threshold optimization (improves recall/precision balance)

Balanced classification performance

Deployment-ready architecture

📌 Future Improvements

Fine-tune CNN instead of fixed embeddings

Add Grad-CAM visualization

Docker containerization

Deploy to cloud (Render / Railway)

Add batch inference API

Add PDF document support

📜 License

MIT License

👨‍💻 Author

Developed as an applied Machine Learning + Computer Vision project demonstrating hybrid modeling and deployment.


---

# 🔥 Optional: Add Badges (Makes It Look Premium)

Add this at the top under the title:

```markdown
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen)

This README now looks:

Research-ready

Internship-ready

Portfolio-ready

Recruiter-friendly

Clean and professional

