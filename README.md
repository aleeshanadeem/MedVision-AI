# 🩺 MedVision AI: Enterprise Clinical Triage Suite

> **ML Empowerment Build Challenge 3.0 Project**  
> *Bridging primary care and rapid symptom triage through advanced machine learning and instant vitals analytics.*

---

<p align="center">
  <img src="![Uploading MedVision AI_ Future of Clinical Care.png…]()
">
</p>

---

## 🚀 Overview
**MedVision AI** is an enterprise-grade web application designed to provide instant, data-driven health risk assessments. Built with a robust Scikit-Learn machine learning pipeline and an intuitive Streamlit interface, it empowers clinicians and patients to perform rapid clinical triage, track session histories, analyze vitals against safe thresholds, and receive customized medical recommendations.

---

## ✨ Key Features
- **🔍 Real-Time Triage Assessment:** Instant risk classification (Safe vs. High Risk) with model confidence probability and optimized inference latency (~11 ms).
- **📊 Vitals vs. Benchmark Analytics:** Visual comparison of patient inputs against standard medical safety thresholds.
- **📁 Patient History Logging:** Session-based tracking to manage multiple patient evaluations seamlessly.
- **💡 Customized Clinical Recommendations:** Actionable diagnostic and lifestyle guidelines generated dynamically based on prediction results.
- **🎨 Professional UI/UX:** Clean, patient-facing enterprise layout featuring the signature **(2AN)K AI** branding badge.

---

## 🛠️ Technical Architecture
- **Core Algorithm:** Random Forest Classifier (optimized ensemble of decision trees).
- **Preprocessing Pipeline:** Scikit-Learn `StandardScaler` for robust feature normalization.
- **Model Serialization:** Saved and deployed via Python's `joblib`.
- **Frontend & Deployment:** Built with **Streamlit** and hosted live on Streamlit Community Cloud.

---

## 🗂️ Project Directory Structure
```text
MedVision-AI/
│
├── app.py                  # Main Streamlit web application
├── medvision_model.pkl     # Trained Random Forest classifier model
├── medvision_scaler.pkl    # Fitted StandardScaler pipeline object
├── requirements.txt        # Cloud deployment dependencies
└── README.md               # Project documentation
